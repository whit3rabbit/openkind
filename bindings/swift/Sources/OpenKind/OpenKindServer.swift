#if os(macOS)
import Darwin
import Foundation

public enum ServerError: Error {
    case invalidConfiguration(String)
    case addressUnavailable(String)
    case alreadyStarted
    case launchFailed(String)
    case exitedBeforeReady(Int32)
    case readinessTimedOut
}

/// Owns one local openkindd process and its HTTP client.
public final class OpenKindServer {
    public let binary: String
    public let httpAddress: String
    public let models: [String]
    public let apiKey: String?
    public let startupTimeout: TimeInterval
    public let shutdownTimeout: TimeInterval
    public let extraArguments: [String]

    private var process: Process?

    public var running: Bool { process?.isRunning == true }

    public var client: OpenKindClient {
        OpenKindClient(baseURL: URL(string: "http://\(httpAddress)")!, apiKey: apiKey)
    }

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

    public func start() async throws {
        guard process == nil else { throw ServerError.alreadyStarted }
        let port = UInt16(httpAddress.split(separator: ":")[1])!
        guard Self.portIsFree(port) else { throw ServerError.addressUnavailable(httpAddress) }

        let child = Process()
        child.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        child.arguments = [binary] + extraArguments + [
            "--http-addr", httpAddress, "--grpc-addr", "0", "--models", models.joined(separator: ","),
        ]
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "OPENKIND_API_KEY")
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

        let deadline = Date().addingTimeInterval(startupTimeout)
        while Date() < deadline {
            if !child.isRunning {
                process = nil
                throw ServerError.exitedBeforeReady(child.terminationStatus)
            }
            do {
                var request = URLRequest(url: URL(string: "http://\(httpAddress)/health")!)
                request.timeoutInterval = 0.25
                let (data, response) = try await URLSession.shared.data(for: request)
                if let http = response as? HTTPURLResponse, http.statusCode == 200,
                   try JSONDecoder().decode(Health.self, from: data).status == "ok", child.isRunning {
                    return
                }
            } catch {
                // The HTTP listener may still be starting.
            }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        await stop()
        throw ServerError.readinessTimedOut
    }

    public func stop() async {
        guard let child = process else { return }
        process = nil
        guard child.isRunning else { return }
        Darwin.kill(child.processIdentifier, SIGTERM)
        let deadline = Date().addingTimeInterval(shutdownTimeout)
        while child.isRunning && Date() < deadline {
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        if child.isRunning {
            Darwin.kill(child.processIdentifier, SIGKILL)
            while child.isRunning { try? await Task.sleep(nanoseconds: 50_000_000) }
        }
    }

    private static func portIsFree(_ port: UInt16) -> Bool {
        let descriptor = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { return false }
        defer { Darwin.close(descriptor) }
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
