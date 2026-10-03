import Foundation
import XCTest
@testable import OpenKind
#if os(macOS)
    private func writeExecutable(_ source: String, in directory: URL) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let binary = directory.appendingPathComponent("fixture")
        try Data(source.utf8).write(to: binary)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: binary.path)
        return binary
    }

import Darwin
#endif

private final class StubProtocol: URLProtocol {
    static var mode = "good"
    static var timeouts: [String: TimeInterval] = [:]
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let path = request.url!.path
        Self.timeouts[path] = request.timeoutInterval
        var status = 200
        var payload: String
        switch path {
        case "/v1/systemone":
            if request.value(forHTTPHeaderField: "Authorization") != "Bearer test-key" {
                status = 401
                payload = #"{"error":{"code":"unauthorized","message":"no token"}}"#
            } else if Self.mode == "rate" {
                status = 429
                payload = #"{"error":{"code":"rate_limited","message":"slow down"}}"#
            } else {
                let selected = Self.mode == "bad" ? "outside" : "billing"
                payload = """
                {"model":"mock","answers":{"team":{"type":"choice","choice":"\(selected)","probabilities":{"billing":0.75,"sales":0.25},"confidence":0.75}},"usage":{"input_tokens":2,"output_tokens":1}}
                """
            }
        case "/v1/models":
            payload = #"{"models":[{"name":"mock","description":"Mock engine","release_date":"2026-01-01"}]}"#
        case "/health":
            payload = request.value(forHTTPHeaderField: "Authorization") == nil ? #"{"status":"ok"}"# : #"{"status":"unexpected auth"}"#
        case "/playground/api/models":
            if request.httpMethod == "POST" {
                if request.value(forHTTPHeaderField: "x-openkind-playground") == "1" {
                    payload = #"{"ok":true}"#
                } else {
                    status = 401
                    payload = #"{"error":{"code":"unauthorized","message":"missing local header"}}"#
                }
            } else {
                payload = #"{"models":[{"name":"mock","description":"Mock","source":"mock","loaded":true,"manageable":true}]}"#
            }
        default:
            status = 404
            payload = #"{"error":{"code":"not_found","message":"missing"}}"#
        }
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil,
                                       headerFields: ["x-typesafe-request-id": "request-123", "Content-Type": "application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(payload.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

final class OpenKindTests: XCTestCase {
    private func makeClient(timeout: TimeInterval? = nil) -> OpenKindClient {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubProtocol.self]
        return OpenKindClient(baseURL: URL(string: "http://example.test")!, apiKey: "test-key",
                              timeout: timeout, session: URLSession(configuration: config))
    }

    func testScoreAnswersPreserveTheRequestedRubricAndExpectation() throws {
        let request = SystemRequest(state: .string("ticket"), model: "mock", questions: [
            "severity": .score(instructions: .string("Rate severity"), criteria: ["low", "high"]),
        ])
        func response(legend: [String: String], probabilities: [String: Double], score: Double) -> SystemResponse {
            SystemResponse(model: "mock", answers: [
                "severity": .score(value: score, legend: legend, probabilities: probabilities, confidence: 0.8),
            ], usage: Usage(input_tokens: 1, output_tokens: 1))
        }
        let legend = ["0": "low", "1": "high"]
        let probabilities = ["0": 0.25, "1": 0.75]
        XCTAssertNoThrow(try validate(response(legend: legend, probabilities: probabilities, score: 0.75), for: request))
        // Matching maps alone do not prove that the server preserved the submitted levels.
        let invalid = [
            response(legend: ["0": "low"], probabilities: ["0": 1], score: 0),
            response(legend: ["0": "high", "1": "low"], probabilities: probabilities, score: 0.75),
            response(legend: ["0": "low", "100": "high"], probabilities: ["0": 0.25, "100": 0.75], score: 0.75),
            response(legend: ["00": "low", "01": "high"], probabilities: ["00": 0.25, "01": 0.75], score: 0.75),
            response(legend: legend, probabilities: probabilities, score: 1),
        ]
        for answer in invalid {
            XCTAssertThrowsError(try validate(answer, for: request)) { error in
                guard case ClientError.invalidResponse = error else { return XCTFail("unexpected error: \(error)") }
            }
        }
    }

    func testExplicitTimeoutAppliesToEvaluationAndModelLoading() async throws {
        StubProtocol.mode = "good"
        for timeout in [nil, TimeInterval(0.75)] {
            let client = makeClient(timeout: timeout)
            _ = try await client.health()
            XCTAssertEqual(StubProtocol.timeouts["/health"], timeout ?? 10)
            _ = try await client.systemOne(state: .string("ticket"), questions: [
                "team": .choice(instructions: .string("Which team?"), criteria: ["billing": nil, "sales": nil]),
            ], model: "mock")
            XCTAssertEqual(StubProtocol.timeouts["/v1/systemone"], timeout ?? 600)
            try await client.setLocalModelLoaded("mock", loaded: true)
            XCTAssertEqual(StubProtocol.timeouts["/playground/api/models"], timeout ?? 600)
        }
    }

    func testEndpointsAndRequestBoundValidation() async throws {
        StubProtocol.mode = "good"
        let client = makeClient()
        let question = Question.choice(instructions: .string("Which team?"), criteria: ["billing": nil, "sales": nil])
        let result = try await client.systemOne(state: .string("good"), questions: ["team": question], model: "mock")
        XCTAssertEqual(result.requestID, "request-123")
        if case .choice(let selected, _, _) = result.data.answers["team"] {
            XCTAssertEqual(selected, "billing")
        } else {
            XCTFail("expected Choice answer")
        }
        let models = try await client.listModels()
        let health = try await client.health()
        XCTAssertEqual(models.data.models.first?.name, "mock")
        XCTAssertEqual(health.data.status, "ok")

        StubProtocol.mode = "bad"
        do {
            _ = try await client.systemOne(state: .string("bad"), questions: ["team": question], model: "mock")
            XCTFail("accepted a choice outside the request criteria")
        } catch ClientError.invalidResponse { }

        StubProtocol.mode = "rate"
        do {
            _ = try await client.systemOne(state: .string("rate"), questions: ["team": question], model: "mock")
            XCTFail("accepted a rate limit error")
        } catch let error as ApiError {
            XCTAssertEqual(error.status, 429)
            XCTAssertEqual(error.code, "rate_limited")
            XCTAssertEqual(error.requestID, "request-123")
        }
    }

    func testRawRequestDecodesAndLocalModelControls() async throws {
        StubProtocol.mode = "good"
        let raw = #"{"state":{"message":"hello"},"model":"mock","questions":{"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":null,"sales":null}}}}"#
        let request = try JSONDecoder().decode(SystemRequest.self, from: Data(raw.utf8))
        let client = makeClient()
        let result = try await client.evaluate(request)
        XCTAssertEqual(result.requestID, "request-123")
        let pretty = JSONEncoder()
        pretty.outputFormatting = [.prettyPrinted, .sortedKeys]
        XCTAssertTrue(String(decoding: try pretty.encode(result.data), as: UTF8.self).contains("billing"))
        let models = try await client.listLocalModels()
        XCTAssertEqual(models.data.models.first?.name, "mock")
        XCTAssertTrue(models.data.models.first?.manageable == true)
        try await client.setLocalModelLoaded("mock", loaded: false)
    }

    func testLiveDaemonWhenConfigured() async throws {
        guard let rawURL = ProcessInfo.processInfo.environment["OPENKIND_TEST_URL"],
              let url = URL(string: rawURL) else { return }
        let client = OpenKindClient(baseURL: url, apiKey: ProcessInfo.processInfo.environment["OPENKIND_TEST_API_KEY"])
        let result = try await client.systemOne(
            state: .string("A customer was charged twice."),
            questions: [
                "billing": .noul(instructions: .string("Is this a billing issue?")),
                "team": .choice(instructions: .string("Which team?"), criteria: ["billing": nil, "sales": nil]),
                "severity": .score(instructions: .string("Rate severity"), criteria: ["low", "high"]),
            ],
            model: "mock"
        )
        XCTAssertEqual(result.data.model, "mock")
        XCTAssertNotNil(result.requestID)
        if case .noul(let value) = result.data.answers["billing"] {
            XCTAssertTrue((0...1).contains(value))
        } else {
            XCTFail("expected Noul answer")
        }
        if case .choice(let selected, _, _) = result.data.answers["team"] {
            XCTAssertTrue(["billing", "sales"].contains(selected))
        } else {
            XCTFail("expected Choice answer")
        }
        if case .score(let value, _, _, _) = result.data.answers["severity"] {
            XCTAssertTrue((0...1).contains(value))
        } else {
            XCTFail("expected Score answer")
        }
        let health = try await client.health()
        XCTAssertEqual(health.data.status, "ok")
        let models = try await client.listModels()
        XCTAssertTrue(models.data.models.contains(where: { $0.name == "mock" }))
    }

#if os(macOS)
    private func freePort() throws -> UInt16 {
        let descriptor = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { throw ServerError.addressUnavailable("socket") }
        defer { Darwin.close(descriptor) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr = in_addr(s_addr: in_addr_t(0x7F000001).bigEndian)
        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else { throw ServerError.addressUnavailable("bind") }
        let sized = withUnsafeMutablePointer(to: &address) { pointer -> Int32 in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { socketAddress in
                var length = socklen_t(MemoryLayout<sockaddr_in>.size)
                return Darwin.getsockname(descriptor, socketAddress, &length)
            }
        }
        guard sized == 0 else { throw ServerError.addressUnavailable("getsockname") }
        return UInt16(bigEndian: address.sin_port)
    }

    func testServerProcessLifecycleAndClient() async throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("test/fake_openkindd.py")
        let server = try OpenKindServer(binary: fixture.path,
                                        httpAddress: "127.0.0.1:\(freePort())",
                                        models: ["mock"], apiKey: "secret")
        do {
            try await server.start()
        } catch {
            print("Fixture startup failed: \(error); last health probe: \(server.lastReadinessFailure ?? "none")")
            throw error
        }
        do {
            XCTAssertTrue(server.running)
            let health = try await server.client.health()
            XCTAssertEqual(health.data.status, "ok")
            let models = try await server.client.listModels()
            XCTAssertEqual(models.data.models.first?.name, "mock")
            let result = try await server.client.systemOne(
                state: .string("billing ticket"),
                questions: [
                    "billing": .noul(instructions: .string("Is this billing?")),
                    "team": .choice(instructions: .string("Which team?"), criteria: ["billing": nil, "sales": nil]),
                    "severity": .score(instructions: .string("Rate severity"), criteria: ["low", "high"]),
                ],
                model: "mock"
            )
            XCTAssertEqual(Set(result.data.answers.keys), Set(["billing", "team", "severity"]))
            XCTAssertEqual(result.requestID, "fixture-request")
            let unauthorized = OpenKindClient(baseURL: server.client.baseURL)
            do {
                _ = try await unauthorized.listModels()
                XCTFail("accepted an unauthenticated models request")
            } catch let error as ApiError {
                XCTAssertEqual(error.status, 401)
            }
        } catch {
            await server.stop()
            throw error
        }
        await server.stop()
        XCTAssertFalse(server.running)
    }

    func testServerReportsHealthFailureAndStopsTimedOutChild() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let binary = directory.appendingPathComponent("no-listener.sh")
        try Data("#!/bin/sh\nexec /bin/sleep 10\n".utf8).write(to: binary)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: binary.path)
        let server = try OpenKindServer(binary: binary.path,
                                        httpAddress: "127.0.0.1:\(freePort())",
                                        models: ["mock"], startupTimeout: 0.3, shutdownTimeout: 0.2)
        do {
            try await server.start()
            XCTFail("started a child without a health listener")
        } catch ServerError.readinessTimedOut {
            XCTAssertNotNil(server.lastReadinessFailure)
        }
        XCTAssertFalse(server.running)
    }

    func testCanceledStartupStopsItsOwnedChildPromptly() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let binary = try writeExecutable("#!/bin/sh\nexec /bin/sleep 10\n", in: directory)
        let server = try OpenKindServer(binary: binary.path,
                                        httpAddress: "127.0.0.1:\(freePort())",
                                        models: ["mock"], startupTimeout: 5, shutdownTimeout: 0.2)
        let startup = Task { try await server.start() }
        let deadline = ProcessInfo.processInfo.systemUptime + 1
        while !server.running && ProcessInfo.processInfo.systemUptime < deadline {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertTrue(server.running)
        let canceledAt = ProcessInfo.processInfo.systemUptime
        startup.cancel()
        do {
            try await startup.value
            XCTFail("canceled startup reported readiness")
        } catch is CancellationError { }
        XCTAssertLessThan(ProcessInfo.processInfo.systemUptime - canceledAt, 1)
        XCTAssertFalse(server.running)
    }

    func testShutdownKeepsOwnershipUntilTheChildHasExited() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let marker = directory.appendingPathComponent("ready")
        let binary = try writeExecutable("""
        #!/bin/sh
        trap '' TERM
        /usr/bin/touch '\(marker.path)'
        exec /bin/sleep 10

        """, in: directory)
        let server = try OpenKindServer(binary: binary.path,
                                        httpAddress: "127.0.0.1:\(freePort())",
                                        models: ["mock"], startupTimeout: 5, shutdownTimeout: 0.2)
        let startup = Task { try await server.start() }
        let deadline = ProcessInfo.processInfo.systemUptime + 1
        while !FileManager.default.fileExists(atPath: marker.path) && ProcessInfo.processInfo.systemUptime < deadline {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: marker.path))
        // SIGTERM is ignored so restart would overlap two children unless ownership is retained.
        server.terminate()
        do {
            try await server.start()
            XCTFail("allowed restart before the old child exited")
        } catch ServerError.alreadyStarted { }
        await server.stop()
        do {
            try await startup.value
            XCTFail("stopped startup reported readiness")
        } catch ServerError.startupStopped { }
        XCTAssertFalse(server.running)
    }

    func testServerScrubsTheLegacyInheritedAPIKey() async throws {
        let key = "OPENDECISION_API_KEY"
        let inherited = ProcessInfo.processInfo.environment[key]
        setenv(key, "inherited-test-key", 1)
        defer {
            if let inherited { setenv(key, inherited, 1) }
            else { unsetenv(key) }
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("test/fake_openkindd.py")
        let binary = try writeExecutable("""
        #!/bin/sh
        if [ "${OPENDECISION_API_KEY+present}" = present ]; then exit 23; fi
        exec '\(fixture.path)' "$@"

        """, in: directory)
        let server = try OpenKindServer(binary: binary.path,
                                        httpAddress: "127.0.0.1:\(freePort())", models: ["mock"])
        try await server.start()
        await server.stop()
        XCTAssertFalse(server.running)
    }

    func testHTTPRedirectsAreReturnedAsErrors() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let binary = try writeExecutable("""
        #!/usr/bin/python3
        import argparse, json, signal, threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        parser = argparse.ArgumentParser()
        parser.add_argument('--http-addr')
        args, _ = parser.parse_known_args()
        host, port = args.http_addr.split(':')
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_GET(self):
                if self.path == '/v1/models':
                    self.send_response(302)
                    self.send_header('Location', '/redirected')
                    self.send_header('x-typesafe-request-id', 'redirect-request')
                    self.end_headers()
                    return
                data = json.dumps({'status':'ok'} if self.path == '/health' else {'models':[]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        server = ThreadingHTTPServer((host, int(port)), Handler)
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
        server.serve_forever(poll_interval=0.05)
        server.server_close()

        """, in: directory)
        let server = try OpenKindServer(binary: binary.path,
                                        httpAddress: "127.0.0.1:\(freePort())", models: ["mock"], apiKey: "secret")
        try await server.start()
        do {
            _ = try await server.client.listModels()
            XCTFail("followed an HTTP redirect")
        } catch let error as ApiError {
            XCTAssertEqual(error.status, 302)
            XCTAssertEqual(error.requestID, "redirect-request")
        } catch {
            await server.stop()
            throw error
        }
        await server.stop()
    }

    func testServerRefusesOccupiedAddress() async throws {
        let descriptor = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { throw ServerError.addressUnavailable("socket") }
        defer { Darwin.close(descriptor) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr = in_addr(s_addr: in_addr_t(0x7F000001).bigEndian)
        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else { throw ServerError.addressUnavailable("bind") }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let named = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.getsockname(descriptor, $0, &length)
            }
        }
        guard named == 0 else { throw ServerError.addressUnavailable("getsockname") }
        let port = UInt16(bigEndian: address.sin_port)
        let server = try OpenKindServer(httpAddress: "127.0.0.1:\(port)")
        do {
            try await server.start()
            XCTFail("started on an occupied port")
        } catch ServerError.addressUnavailable { }
        XCTAssertFalse(server.running)
    }

    func testServerStartsRealBinaryWhenConfigured() async throws {
        guard let binary = ProcessInfo.processInfo.environment["OPENKIND_TEST_BINARY"] else { return }
        let server = try OpenKindServer(binary: binary,
                                        httpAddress: "127.0.0.1:\(freePort())",
                                        models: ["mock"], apiKey: "dev-key",
                                        extraArguments: ["--playground", "on"])
        try await server.start()
        do {
            let health = try await server.client.health()
            XCTAssertEqual(health.data.status, "ok")
            let result = try await server.client.systemOne(
                state: .string("billing ticket"),
                questions: ["billing": .noul(instructions: .string("Is this billing?"))],
                model: "mock"
            )
            if case .some(.noul) = result.data.answers["billing"] { }
            else { XCTFail("expected Noul answer") }
            let local = try await server.client.listLocalModels()
            XCTAssertTrue(local.data.models.contains(where: { $0.name == "mock" && $0.manageable && $0.loaded }))
            try await server.client.setLocalModelLoaded("mock", loaded: false)
            let unloaded = try await server.client.listLocalModels()
            XCTAssertTrue(unloaded.data.models.contains(where: {
                $0.name == "mock" && !$0.loaded
            }))
            try await server.client.setLocalModelLoaded("mock", loaded: true)
        } catch {
            await server.stop()
            throw error
        }
        await server.stop()
        XCTAssertFalse(server.running)
    }
#endif
}
