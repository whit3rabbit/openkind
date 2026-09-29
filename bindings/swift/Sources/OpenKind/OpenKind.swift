import Foundation

public indirect enum JSONValue: Codable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        if value.decodeNil() { self = .null }
        else if let bool = try? value.decode(Bool.self) { self = .bool(bool) }
        else if let number = try? value.decode(Double.self) { self = .number(number) }
        else if let string = try? value.decode(String.self) { self = .string(string) }
        else if let array = try? value.decode([JSONValue].self) { self = .array(array) }
        else { self = .object(try value.decode([String: JSONValue].self)) }
    }

    public func encode(to encoder: Encoder) throws {
        var value = encoder.singleValueContainer()
        switch self {
        case .null: try value.encodeNil()
        case .bool(let item): try value.encode(item)
        case .number(let item): try value.encode(item)
        case .string(let item): try value.encode(item)
        case .array(let item): try value.encode(item)
        case .object(let item): try value.encode(item)
        }
    }
}

public struct NoulCriteria: Codable {
    public let `true`: String
    public let `false`: String

    public init(true: String, false: String) {
        self.true = `true`
        self.false = `false`
    }
}

public enum Question: Codable {
    case noul(instructions: JSONValue, criteria: NoulCriteria? = nil)
    case choice(instructions: JSONValue, criteria: [String: String?])
    case score(instructions: JSONValue, criteria: [String])

    private enum CodingKeys: String, CodingKey { case type, instructions, criteria }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .noul(let instructions, let criteria):
            try container.encode("noul", forKey: .type)
            try container.encode(instructions, forKey: .instructions)
            try container.encodeIfPresent(criteria, forKey: .criteria)
        case .choice(let instructions, let criteria):
            try container.encode("choice", forKey: .type)
            try container.encode(instructions, forKey: .instructions)
            try container.encode(criteria, forKey: .criteria)
        case .score(let instructions, let criteria):
            try container.encode("score", forKey: .type)
            try container.encode(instructions, forKey: .instructions)
            try container.encode(criteria, forKey: .criteria)
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let instructions = try container.decode(JSONValue.self, forKey: .instructions)
        switch try container.decode(String.self, forKey: .type) {
        case "noul":
            self = .noul(instructions: instructions, criteria: try container.decodeIfPresent(NoulCriteria.self, forKey: .criteria))
        case "choice":
            self = .choice(instructions: instructions, criteria: try container.decode([String: String?].self, forKey: .criteria))
        case "score":
            self = .score(instructions: instructions, criteria: try container.decode([String].self, forKey: .criteria))
        default:
            throw DecodingError.dataCorruptedError(forKey: .type, in: container, debugDescription: "unknown question type")
        }
    }
}

public struct SystemRequest: Codable {
    public let state: JSONValue
    public let model: String
    public let questions: [String: Question]

    public init(state: JSONValue, model: String, questions: [String: Question]) {
        self.state = state
        self.model = model
        self.questions = questions
    }
}

public enum Answer: Codable {
    case noul(Double)
    case choice(selected: String, probabilities: [String: Double], confidence: Double)
    case score(value: Double, legend: [String: String], probabilities: [String: Double], confidence: Double)

    private enum CodingKeys: String, CodingKey { case type, noul, choice, score, legend, probabilities, confidence }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(String.self, forKey: .type) {
        case "noul":
            self = .noul(try container.decode(Double.self, forKey: .noul))
        case "choice":
            self = .choice(
                selected: try container.decode(String.self, forKey: .choice),
                probabilities: try container.decode([String: Double].self, forKey: .probabilities),
                confidence: try container.decode(Double.self, forKey: .confidence)
            )
        case "score":
            self = .score(
                value: try container.decode(Double.self, forKey: .score),
                legend: try container.decode([String: String].self, forKey: .legend),
                probabilities: try container.decode([String: Double].self, forKey: .probabilities),
                confidence: try container.decode(Double.self, forKey: .confidence)
            )
        default:
            throw DecodingError.dataCorruptedError(forKey: .type, in: container, debugDescription: "unknown answer type")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .noul(let value):
            try container.encode("noul", forKey: .type)
            try container.encode(value, forKey: .noul)
        case .choice(let selected, let probabilities, let confidence):
            try container.encode("choice", forKey: .type)
            try container.encode(selected, forKey: .choice)
            try container.encode(probabilities, forKey: .probabilities)
            try container.encode(confidence, forKey: .confidence)
        case .score(let value, let legend, let probabilities, let confidence):
            try container.encode("score", forKey: .type)
            try container.encode(value, forKey: .score)
            try container.encode(legend, forKey: .legend)
            try container.encode(probabilities, forKey: .probabilities)
            try container.encode(confidence, forKey: .confidence)
        }
    }
}

public struct Usage: Codable {
    public let input_tokens: UInt32
    public let output_tokens: UInt32
}

public struct SystemResponse: Codable {
    public let model: String
    public let answers: [String: Answer]
    public let usage: Usage
}

public struct ModelMetadata: Decodable {
    public let name: String
    public let description: String
    public let release_date: String
}

public struct ModelsResponse: Decodable { public let models: [ModelMetadata] }
public struct Health: Decodable { public let status: String }

public struct LocalModel: Decodable, Identifiable {
    public let name: String
    public let description: String
    public let source: String
    public let loaded: Bool
    public let manageable: Bool
    public var id: String { name }
}

public struct LocalModelsResponse: Decodable { public let models: [LocalModel] }
private struct LocalModelActionResponse: Decodable { let ok: Bool }

public struct ApiResult<Value> {
    public let data: Value
    public let requestID: String?
}

public struct ApiError: Error {
    public let status: Int
    public let code: String?
    public let message: String
    public let requestID: String?
}

public enum ClientError: Error {
    case invalidResponse(String)
    case nonHTTPResponse
}

private struct ErrorEnvelope: Decodable {
    struct Details: Decodable { let code: String?; let message: String? }
    let error: Details
}

public final class OpenKindClient {
    public let baseURL: URL
    public let defaultModel: String
    private let apiKey: String?
    private let timeout: TimeInterval
    private let session: URLSession

    public init(
        baseURL: URL = URL(string: "http://127.0.0.1:18080")!,
        apiKey: String? = nil,
        defaultModel: String = "jev-latest",
        timeout: TimeInterval = 10,
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.defaultModel = defaultModel
        self.timeout = timeout
        self.session = session
    }

    private func send<Value: Decodable>(
        _ path: String, method: String, body: Data? = nil, localModelAction: Bool = false,
        requestTimeout: TimeInterval? = nil
    ) async throws -> ApiResult<Value> {
        let relative = path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        var request = URLRequest(url: baseURL.appendingPathComponent(relative))
        request.httpMethod = method
        request.timeoutInterval = requestTimeout ?? timeout
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        if path != "/health", let apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        if localModelAction {
            request.setValue("1", forHTTPHeaderField: "x-openkind-playground")
        }
        let (data, urlResponse) = try await session.data(for: request)
        guard let response = urlResponse as? HTTPURLResponse else { throw ClientError.nonHTTPResponse }
        let requestID = response.value(forHTTPHeaderField: "x-typesafe-request-id")
        guard (200..<300).contains(response.statusCode) else {
            let details = try? JSONDecoder().decode(ErrorEnvelope.self, from: data)
            throw ApiError(status: response.statusCode, code: details?.error.code,
                           message: details?.error.message ?? "HTTP \(response.statusCode)", requestID: requestID)
        }
        let decoded = try JSONDecoder().decode(Value.self, from: data)
        return ApiResult(data: decoded, requestID: requestID)
    }

    public func evaluate(_ request: SystemRequest) async throws -> ApiResult<SystemResponse> {
        let result: ApiResult<SystemResponse> = try await send(
            "/v1/systemone", method: "POST", body: JSONEncoder().encode(request),
            requestTimeout: 600)
        try validate(result.data, for: request)
        return result
    }

    public func systemOne(state: JSONValue, questions: [String: Question], model: String? = nil) async throws -> ApiResult<SystemResponse> {
        try await evaluate(SystemRequest(state: state, model: model ?? defaultModel, questions: questions))
    }

    public func listModels() async throws -> ApiResult<ModelsResponse> {
        try await send("/v1/models", method: "GET")
    }

    public func health() async throws -> ApiResult<Health> {
        try await send("/health", method: "GET")
    }

    /// Local daemon controls are outside the public TypeSafe wire contract.
    public func listLocalModels() async throws -> ApiResult<LocalModelsResponse> {
        try await send("/playground/api/models", method: "GET")
    }

    public func setLocalModelLoaded(_ name: String, loaded: Bool) async throws {
        let body = try JSONSerialization.data(withJSONObject: ["name": name, "loaded": loaded])
        let result: ApiResult<LocalModelActionResponse> = try await send(
            "/playground/api/models", method: "POST", body: body,
            localModelAction: true, requestTimeout: 600)
        guard result.data.ok else { throw ClientError.invalidResponse("model action was not applied") }
    }
}

private func probability(_ value: Double, _ label: String) throws {
    guard value.isFinite && (0...1).contains(value) else { throw ClientError.invalidResponse("\(label) must be a probability") }
}

private func distribution(_ values: [String: Double], _ label: String) throws {
    for (key, value) in values { try probability(value, "\(label).\(key)") }
    guard abs(values.values.reduce(0, +) - 1) <= 1e-3 else {
        throw ClientError.invalidResponse("\(label) must sum to one")
    }
}

public func validate(_ response: SystemResponse, for request: SystemRequest) throws {
    guard Set(response.answers.keys) == Set(request.questions.keys) else {
        throw ClientError.invalidResponse("answer IDs do not match question IDs")
    }
    for (id, question) in request.questions {
        guard let answer = response.answers[id] else { throw ClientError.invalidResponse("missing answer for \(id)") }
        switch (question, answer) {
        case (.noul, .noul(let value)):
            try probability(value, "\(id).noul")
        case (.choice(_, let criteria), .choice(let selected, let values, let confidence)):
            guard criteria.keys.contains(selected), Set(criteria.keys) == Set(values.keys) else {
                throw ClientError.invalidResponse("choice keys mismatch for \(id)")
            }
            try distribution(values, "\(id).probabilities")
            try probability(confidence, "\(id).confidence")
        case (.score, .score(let score, let legend, let values, let confidence)):
            guard Set(legend.keys) == Set(values.keys),
                  legend.keys.allSatisfy({ !$0.isEmpty && $0.allSatisfy(\.isASCIIDigit) }),
                  let maxIndex = legend.keys.compactMap(UInt32.init).max(),
                  score.isFinite && score >= 0 && score <= Double(maxIndex) else {
                throw ClientError.invalidResponse("score legend or value mismatch for \(id)")
            }
            try distribution(values, "\(id).probabilities")
            try probability(confidence, "\(id).confidence")
        default:
            throw ClientError.invalidResponse("answer type mismatch for \(id)")
        }
    }
}

private extension Character {
    var isASCIIDigit: Bool { unicodeScalars.allSatisfy { $0.value >= 48 && $0.value <= 57 } }
}
