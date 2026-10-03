#if os(macOS)
import Darwin
import Foundation

/// Errors occurring during local openkindd daemon lifecycle management.
public enum ServerError: Error {
    /// Configuration option value was invalid.
    case invalidConfiguration(String)
    /// Target HTTP loopback address is already occupied by another listener.
    case addressUnavailable(String)
    /// A server process has already been started by this instance.
    case alreadyStarted
    /// Failed to launch the child process executable.
    case launchFailed(String)
    /// The process exited prematurely before passing the health check.
    case exitedBeforeReady(Int32)
    /// Startup timed out waiting for the health check to succeed.
    case readinessTimedOut
    /// The owned child was stopped while startup was waiting for readiness.
    case startupStopped
}

/// Owns one local openkindd process and its HTTP client.
public final class OpenKindServer: @unchecked Sendable {
    /// Binary executable name or path to spawn.
    public let binary: String
    /// Local loopback address (`127.0.0.1:<port>`) for HTTP listener.
    public let httpAddress: String
    /// Model aliases loaded into the daemon.
    public let models: [String]
    /// Optional bearer API key configured for requests.
    public let apiKey: String?
    /// Maximum time in seconds to wait for successful health check.
    public let startupTimeout: TimeInterval
    /// Maximum time in seconds to wait for clean SIGTERM shutdown.
    public let shutdownTimeout: TimeInterval
    /// Additional command-line flags forwarded to the child process.
    public let extraArguments: [String]

    // Synchronous configuration and all mutable lifecycle state share this lock.
    private let lifecycleLock = NSLock()
    private var process: Process?
    private var stopping = false
    // Keep a bounded diagnostic for tests without recording response bodies or credentials.
    private var readinessFailure: String?
    var lastReadinessFailure: String? { withLifecycleLock { readinessFailure } }

    /// True if the child process is currently running.
    public var running: Bool { withLifecycleLock { process?.isRunning == true } }

    /// Client preconfigured to communicate with this local daemon.
    public var client: OpenKindClient {
        OpenKindClient(baseURL: URL(string: "http://\(httpAddress)")!, apiKey: apiKey)
    }

    /// Initialize a local daemon manager with lifecycle options.
    public init(
        binary: String = "openkindd",
        httpAddress: String = "127.0.0.1:18080",
        models: [String] = ["mock", "jev-latest"],
        apiKey: String? = nil,
        startupTimeout: TimeInterval = 10,
        shutdownTimeout: TimeInterval = 5,
        extraArguments: [String] = []
    ) throws {
        let parts = httpAddress.split(separator: ":", omittingEmptySubsequences: false)
        guard parts.count == 2, parts[0] == "127.0.0.1",
              let port = UInt16(parts[1]), port > 0 else {
            throw ServerError.invalidConfiguration("httpAddress must use 127.0.0.1 and a nonzero port")
        }
        guard !models.isEmpty, models.allSatisfy({ !$0.isEmpty && !$0.contains(",") }) else {
            throw ServerError.invalidConfiguration("models must contain nonempty aliases without commas")
        }
        if let apiKey {
            guard !apiKey.isEmpty, apiKey.utf8.allSatisfy({ (0x21...0x7e).contains($0) }) else {
                throw ServerError.invalidConfiguration("apiKey must be nonempty visible ASCII without whitespace")
            }
        }
        guard startupTimeout.isFinite, shutdownTimeout.isFinite,
              startupTimeout > 0, shutdownTimeout > 0 else {
            throw ServerError.invalidConfiguration("timeouts must be positive")
        }
        let managed = ["--http-addr", "--grpc-addr", "--models", "--api-key"]
        guard !extraArguments.contains(where: { arg in
            managed.contains(where: { arg == $0 || arg.hasPrefix($0 + "=") })
        }) else {
            throw ServerError.invalidConfiguration("extraArguments cannot override managed server flags")
        }
        self.binary = binary
        self.httpAddress = httpAddress
        self.models = models
        self.apiKey = apiKey
        self.startupTimeout = startupTimeout
        self.shutdownTimeout = shutdownTimeout
        self.extraArguments = extraArguments
    }

    /// Start the child server process and wait until the health endpoint reports readiness.
    public func start() async throws {
        try Task.checkCancellation()
        let child = try withLifecycleLock { try launchChild() }
        do {
            try await waitForReadiness(child)
        } catch {
            // Cleanup must finish even when the caller canceled its startup task.
            await stop(child)
            throw error
        }
    }

    private func withLifecycleLock<Value>(_ body: () throws -> Value) rethrows -> Value {
        lifecycleLock.lock()
        defer { lifecycleLock.unlock() }
        return try body()
    }

    // The caller holds lifecycleLock through launch to reserve ownership across concurrent starts.
    private func launchChild() throws -> Process {
        guard process?.isRunning != true else { throw ServerError.alreadyStarted }
        readinessFailure = nil
        stopping = false
        let port = UInt16(httpAddress.split(separator: ":")[1])!
        guard Self.portIsFree(port) else { throw ServerError.addressUnavailable(httpAddress) }

        let child = Process()
        child.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        child.arguments = [binary] + extraArguments + [
            "--http-addr", httpAddress, "--grpc-addr", "0", "--models", models.joined(separator: ","),
        ]
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "OPENKIND_API_KEY")
        environment.removeValue(forKey: "OPENDECISION_API_KEY")
        environment.removeValue(forKey: "OPENPICK_API_KEY")
        environment.removeValue(forKey: "TYPESAFE_API_KEY")
        if let apiKey { environment["OPENKIND_API_KEY"] = apiKey }
        child.environment = environment
        child.standardInput = FileHandle.nullDevice
        child.standardOutput = FileHandle.nullDevice
        do {
            try child.run()
        } catch {
            throw ServerError.launchFailed(error.localizedDescription)
        }
        process = child
        return child
    }

    private func waitForReadiness(_ child: Process) async throws {
        let deadline = ProcessInfo.processInfo.systemUptime + startupTimeout
        while ProcessInfo.processInfo.systemUptime < deadline {
            try Task.checkCancellation()
            guard withLifecycleLock({ process === child && !stopping }) else {
                throw ServerError.startupStopped
            }
            if !child.isRunning {
                throw ServerError.exitedBeforeReady(child.terminationStatus)
            }
            do {
                var request = URLRequest(url: URL(string: "http://\(httpAddress)/health")!)
                request.timeoutInterval = 0.25
                let (data, response) = try await URLSession.shared.data(for: request, delegate: RedirectPolicy())
                if let http = response as? HTTPURLResponse, http.statusCode == 200,
                   try JSONDecoder().decode(Health.self, from: data).status == "ok",
                   withLifecycleLock({ process === child && !stopping && child.isRunning }) {
                    try Task.checkCancellation()
                    return
                }
                recordReadinessFailure((response as? HTTPURLResponse).map {
                    "health HTTP \($0.statusCode), status did not report ok"
                } ?? "health response was not HTTP", child: child)
            } catch {
                try Task.checkCancellation()
                // The HTTP listener may still be starting.
                let failure = error as NSError
                recordReadinessFailure("\(failure.domain) (\(failure.code))", child: child)
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        throw ServerError.readinessTimedOut
    }

    private func recordReadinessFailure(_ failure: String, child: Process) {
        withLifecycleLock {
            if process === child { readinessFailure = failure }
        }
    }

    /// Gracefully stop the running child process using SIGTERM, escalating to SIGKILL if timeout expires.
    public func stop() async {
        guard let child = withLifecycleLock({ process }) else { return }
        await stop(child)
    }

    private func stop(_ child: Process) async {
        let owned = withLifecycleLock {
            guard process === child else { return false }
            stopping = true
            return true
        }
        guard owned else { return }
        // A canceled startup still needs non-canceled sleeps while its child shuts down.
        let shutdownTimeout = shutdownTimeout
        await Task.detached {
            if child.isRunning { Darwin.kill(child.processIdentifier, SIGTERM) }
            let deadline = ProcessInfo.processInfo.systemUptime + shutdownTimeout
            while child.isRunning && ProcessInfo.processInfo.systemUptime < deadline {
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
            if child.isRunning {
                Darwin.kill(child.processIdentifier, SIGKILL)
                while child.isRunning { try? await Task.sleep(nanoseconds: 50_000_000) }
            }
        }.value
        withLifecycleLock {
            if process === child {
                process = nil
                stopping = false
            }
        }
    }

    /// Synchronous quit hook for an app that is already terminating.
    public func terminate() {
        withLifecycleLock {
            guard let child = process else { return }
            stopping = true
            if child.isRunning { Darwin.kill(child.processIdentifier, SIGTERM) }
        }
    }

    private static func portIsFree(_ port: UInt16) -> Bool {
        let descriptor = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { return false }
        defer { Darwin.close(descriptor) }
        // A closed daemon can leave TIME_WAIT connections; these do not own a listener.
        var reuseAddress: Int32 = 1
        guard Darwin.setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &reuseAddress,
                                socklen_t(MemoryLayout<Int32>.size)) == 0 else { return false }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(port).bigEndian
        address.sin_addr = in_addr(s_addr: in_addr_t(0x7F000001).bigEndian)
        return withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0
            }
        }
    }
}
#endif
