import Foundation

/// Recursive representation of arbitrary JSON-serializable values.
public indirect enum JSONValue: Codable, Sendable {
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

/// Text criteria defining true and false labels for binary noul questions.
public struct NoulCriteria: Codable, Sendable {
    public let `true`: String
    public let `false`: String

    public init(true: String, false: String) {
        self.true = `true`
        self.false = `false`
    }
}

/// Evaluation questions supported by the System One protocol: noul, choice, and score.
public enum Question: Codable, Sendable {
    /// Binary question evaluated to a probability mass between 0.0 and 1.0.
    case noul(instructions: JSONValue, criteria: NoulCriteria? = nil)
    /// Multiple-choice question evaluated over candidate option criteria.
    case choice(instructions: JSONValue, criteria: [String: String?])
    /// Rubric score question evaluated over ordered score levels.
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

/// Top-level System One evaluation request containing state context, model alias, and questions.
public struct SystemRequest: Codable, Sendable {
    public let state: JSONValue
    public let model: String
    public let questions: [String: Question]

    public init(state: JSONValue, model: String, questions: [String: Question]) {
        self.state = state
        self.model = model
        self.questions = questions
    }
}

/// Evaluated answer types matching the requested question types.
public enum Answer: Codable, Sendable {
    /// Evaluated binary decision returning probability mass.
    case noul(Double)
    /// Evaluated multiple-choice decision with selected candidate, probabilities, and confidence.
    case choice(selected: String, probabilities: [String: Double], confidence: Double)
    /// Evaluated rubric score decision with expected value, rubric legend, probabilities, and confidence.
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

/// Token usage counters reported for input and output.
public struct Usage: Codable, Sendable {
    public let input_tokens: UInt32
    public let output_tokens: UInt32
}

/// System One evaluation response containing model name, evaluated answers, and token usage.
public struct SystemResponse: Codable, Sendable {
    public let model: String
    public let answers: [String: Answer]
    public let usage: Usage
}

/// Metadata describing an available model profile in the catalog.
public struct ModelMetadata: Decodable, Sendable {
    public let name: String
    public let description: String
    public let release_date: String
}

/// Response payload from /v1/models listing available models.
public struct ModelsResponse: Decodable, Sendable { public let models: [ModelMetadata] }

/// Daemon health check status response.
public struct Health: Decodable, Sendable { public let status: String }

/// Information describing a locally installed model profile.
public struct LocalModel: Decodable, Identifiable, Sendable {
    public let name: String
    public let description: String
    public let source: String
    public let loaded: Bool
    public let manageable: Bool
    public var id: String { name }
}

/// Response payload listing locally installed model profiles.
public struct LocalModelsResponse: Decodable, Sendable { public let models: [LocalModel] }
private struct LocalModelActionResponse: Decodable { let ok: Bool }

/// Result envelope wrapping decoded response data and optional request ID header.
public struct ApiResult<Value> {
    public let data: Value
    public let requestID: String?
}

extension ApiResult: Sendable where Value: Sendable {}

/// Error thrown when an HTTP call to the daemon fails with a non-2xx status code.
public struct ApiError: Error {
    public let status: Int
    public let code: String?
    public let message: String
    public let requestID: String?
}

/// Client-side validation or transport errors.
public enum ClientError: Error {
    case invalidResponse(String)
    case nonHTTPResponse
}

private struct ErrorEnvelope: Decodable {
    struct Details: Decodable { let code: String?; let message: String? }
    let error: Details
}

final class RedirectPolicy: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        // A configured decision endpoint must not redirect credentials or request state.
        completionHandler(nil)
    }
}

/// Asynchronous HTTP client for interacting with the OpenKind daemon and evaluating decisions.
public final class OpenKindClient: Sendable {
    /// Base URL of the target daemon.
    public let baseURL: URL
    /// Default model alias used when omitted from evaluation requests.
    public let defaultModel: String
    private let apiKey: String?
    private let timeout: TimeInterval
    private let evaluationTimeout: TimeInterval
    private let session: URLSession

    /// Initialize a client. An explicit timeout applies to every request; defaults
    /// are 10 seconds for probes and listings, 600 seconds for evaluation and model loading.
    public init(
        baseURL: URL = URL(string: "http://127.0.0.1:18080")!,
        apiKey: String? = nil,
        defaultModel: String = "jev-latest",
        timeout: TimeInterval? = nil,
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.defaultModel = defaultModel
        self.timeout = timeout ?? 10
        self.evaluationTimeout = timeout ?? 600
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
        let (data, urlResponse) = try await session.data(for: request, delegate: RedirectPolicy())
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

    /// Submit a SystemRequest evaluation payload to `/v1/systemone` and validate the returned answers.
    public func evaluate(_ request: SystemRequest) async throws -> ApiResult<SystemResponse> {
        let result: ApiResult<SystemResponse> = try await send(
            "/v1/systemone", method: "POST", body: JSONEncoder().encode(request),
            requestTimeout: evaluationTimeout)
        try validate(result.data, for: request)
        return result
    }

    /// Convenience method to evaluate questions against state using the default or specified model.
    public func systemOne(state: JSONValue, questions: [String: Question], model: String? = nil) async throws -> ApiResult<SystemResponse> {
        try await evaluate(SystemRequest(state: state, model: model ?? defaultModel, questions: questions))
    }

    /// Query `/v1/models` to retrieve available model profiles.
    public func listModels() async throws -> ApiResult<ModelsResponse> {
        try await send("/v1/models", method: "GET")
    }

    /// Query `/health` to verify daemon availability.
    public func health() async throws -> ApiResult<Health> {
        try await send("/health", method: "GET")
    }

    /// Local daemon controls are outside the public TypeSafe wire contract.
    public func listLocalModels() async throws -> ApiResult<LocalModelsResponse> {
        try await send("/playground/api/models", method: "GET")
    }

    /// Load or unload a local model in the running daemon.
    public func setLocalModelLoaded(_ name: String, loaded: Bool) async throws {
        let body = try JSONSerialization.data(withJSONObject: ["name": name, "loaded": loaded])
        let result: ApiResult<LocalModelActionResponse> = try await send(
            "/playground/api/models", method: "POST", body: body,
            localModelAction: true, requestTimeout: evaluationTimeout)
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

/// Validate that a SystemResponse matches question IDs, types, and probability constraints of the request.
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
        case (.score(_, let criteria), .score(let score, let legend, let values, let confidence)):
            let expectedLegend = Dictionary(uniqueKeysWithValues: criteria.enumerated().map { (String($0.offset), $0.element) })
            let maxIndex = Double(max(0, criteria.count - 1))
            guard !criteria.isEmpty, legend == expectedLegend, Set(legend.keys) == Set(values.keys),
                  score.isFinite && score >= 0 && score <= maxIndex else {
                throw ClientError.invalidResponse("score legend or value mismatch for \(id)")
            }
            try distribution(values, "\(id).probabilities")
            try probability(confidence, "\(id).confidence")
            let expectedScore = criteria.indices.reduce(0.0) { $0 + Double($1) * values[String($1)]! }
            guard abs(score - expectedScore) <= 1e-3 * max(1, maxIndex) else {
                throw ClientError.invalidResponse("score does not match probabilities for \(id)")
            }
        default:
            throw ClientError.invalidResponse("answer type mismatch for \(id)")
        }
    }
}
