import Foundation
import XCTest
@testable import OpenKind
#if os(macOS)
import Darwin
#endif

private final class StubProtocol: URLProtocol {
    static var mode = "good"
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let path = request.url!.path
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
    private func makeClient() -> OpenKindClient {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubProtocol.self]
        return OpenKindClient(baseURL: URL(string: "http://example.test")!, apiKey: "test-key",
                              session: URLSession(configuration: config))
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
        try await server.start()
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
