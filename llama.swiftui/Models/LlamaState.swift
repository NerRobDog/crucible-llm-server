import Foundation

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
}

@MainActor
class LlamaState: ObservableObject {
    @Published var messageLog = ""
    @Published var cacheCleared = false
    @Published var downloadedModels: [Model] = []
    @Published var undownloadedModels: [Model] = []
    let NS_PER_S = 1_000_000_000.0

    @Published var serverRunning = false
    @Published var serverAddress = ""
    let httpServer = HTTPServer(port: 8080)

    // Crucible: observable model-load state, surfaced over HTTP (/v1/models/status)
    enum LoadState: String { case idle, loading, ready, error }
    @Published var loadState: LoadState = .idle
    @Published var loadedModelName: String = ""
    @Published var loadError: String = ""

    private var llamaContext: LlamaContext?
    private var defaultModelUrl: URL? {
        Bundle.main.url(forResource: "ggml-model", withExtension: "gguf", subdirectory: "models")
        // Bundle.main.url(forResource: "llama-2-7b-chat", withExtension: "Q2_K.gguf", subdirectory: "models")
    }

    init() {
        loadModelsFromDisk()
        loadDefaultModels()
    }

    private func loadModelsFromDisk() {
        do {
            let documentsURL = getDocumentsDirectory()
            let modelURLs = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
            for modelURL in modelURLs {
                let modelName = modelURL.deletingPathExtension().lastPathComponent
                downloadedModels.append(Model(name: modelName, url: "", filename: modelURL.lastPathComponent, status: "downloaded"))
            }
        } catch {
            print("Error loading models from disk: \(error)")
        }
    }

    private func loadDefaultModels() {
        loadModel(modelUrl: defaultModelUrl)

        for model in defaultModels {
            let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
            if FileManager.default.fileExists(atPath: fileURL.path) {

            } else {
                var undownloadedModel = model
                undownloadedModel.status = "download"
                undownloadedModels.append(undownloadedModel)
            }
        }
    }

    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
    private let defaultModels: [Model] = [
        Model(
            name: "Gemma-3-4B-IT (Q4_K_M, 2.9 GiB)",
            url: "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf?download=true",
            filename: "gemma-3-4b-it-Q4_K_M.gguf", status: "download"
        ),
        Model(
            name: "Qwen-3.5-4B (Q4_K_M, 2.7 GiB)",
            url: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF/resolve/main/Qwen_Qwen3.5-4B-Q4_K_M.gguf?download=true",
            filename: "qwen3.5-4b-Q4_K_M.gguf", status: "download"
        ),
        Model(name: "TinyLlama-1.1B (Q4_0, 0.6 GiB)",url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_0.gguf?download=true",filename: "tinyllama-1.1b-1t-openorca.Q4_0.gguf", status: "download"),
        Model(
            name: "TinyLlama-1.1B Chat (Q8_0, 1.1 GiB)",
            url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true",
            filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf", status: "download"
        ),

        Model(
            name: "TinyLlama-1.1B (F16, 2.2 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-f16.gguf?download=true",
            filename: "tinyllama-1.1b-f16.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q4_0, 1.6 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q4_0.gguf?download=true",
            filename: "phi-2-q4_0.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q8_0, 2.8 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q8_0.gguf?download=true",
            filename: "phi-2-q8_0.gguf", status: "download"
        ),

        Model(
            name: "Mistral-7B-v0.1 (Q4_0, 3.8 GiB)",
            url: "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_0.gguf?download=true",
            filename: "mistral-7b-v0.1.Q4_0.gguf", status: "download"
        ),
        Model(
            name: "OpenHermes-2.5-Mistral-7B (Q3_K_M, 3.52 GiB)",
            url: "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q3_K_M.gguf?download=true",
            filename: "openhermes-2.5-mistral-7b.Q3_K_M.gguf", status: "download"
        )
    ]
    /// Kicks off a model load on a background executor (create_context is heavy and
    /// otherwise blocks the main thread, freezing the UI and the HTTP server). State
    /// and errors are reported via loadState/loadError/messageLog and CrucibleLog,
    /// never swallowed into print().
    func loadModel(modelUrl: URL?, nCtx: Int32 = 8192) {
        guard let modelUrl else {
            messageLog += "Load a model from the list below\n"
            return
        }
        let name = modelUrl.lastPathComponent
        let path = modelUrl.path(percentEncoded: false)

        // Drop the previous context first so its memory is freed before we allocate
        // the new one (critical on a 12GB device near the jetsam limit).
        llamaContext = nil
        loadState = .loading
        loadError = ""
        loadedModelName = name
        messageLog += "Loading model \(name)...\n"
        CrucibleLog.shared.log("[crucible] load requested: \(name) n_ctx=\(nCtx)")

        Task.detached(priority: .userInitiated) {
            do {
                let ctx = try LlamaContext.create_context(path: path, nCtx: nCtx)
                await MainActor.run {
                    self.llamaContext = ctx
                    self.loadState = .ready
                    self.messageLog += "Loaded model \(name)\n"
                    self.updateDownloadedModels(modelName: name, status: "downloaded")
                }
            } catch {
                await MainActor.run {
                    self.loadState = .error
                    self.loadError = "\(error)"
                    self.messageLog += "ERROR loading \(name): \(error)\n"
                }
            }
        }
    }

    // MARK: - API-driven model management (used by the HTTP server)

    /// Load a model already present in the app's Documents dir, by filename.
    @discardableResult
    func loadModelByName(_ filename: String, nCtx: Int32 = 8192) -> Bool {
        let url = getDocumentsDirectory().appendingPathComponent(filename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            CrucibleLog.shared.log("[crucible] load rejected: file not found: \(filename)")
            return false
        }
        loadModel(modelUrl: url, nCtx: nCtx)
        return true
    }

    func unloadModel() {
        llamaContext = nil
        loadState = .idle
        loadedModelName = ""
        loadError = ""
        messageLog += "Model unloaded\n"
        CrucibleLog.shared.log("[crucible] model unloaded")
    }

    /// Filenames of every model file sitting in Documents (what /v1/models lists).
    func availableModelFiles() -> [String] {
        let dir = getDocumentsDirectory()
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])) ?? []
        return urls.map { $0.lastPathComponent }.sorted()
    }


    private func updateDownloadedModels(modelName: String, status: String) {
        undownloadedModels.removeAll { $0.name == modelName }
    }


    func complete(text: String) async {
        guard let llamaContext else {
            return
        }

        let t_start = DispatchTime.now().uptimeNanoseconds
        await llamaContext.completion_init(text: text)
        let t_heat_end = DispatchTime.now().uptimeNanoseconds
        let t_heat = Double(t_heat_end - t_start) / NS_PER_S

        messageLog += "\(text)"

        Task.detached {
            while await !llamaContext.is_done {
                let result = await llamaContext.completion_loop()
                await MainActor.run {
                    self.messageLog += "\(result)"
                }
            }

            let t_end = DispatchTime.now().uptimeNanoseconds
            let t_generation = Double(t_end - t_heat_end) / self.NS_PER_S
            let tokens_per_second = Double(await llamaContext.n_len) / t_generation

            await llamaContext.clear()

            await MainActor.run {
                self.messageLog += """
                    \n
                    Done
                    Heat up took \(t_heat)s
                    Generated \(tokens_per_second) t/s\n
                    """
            }
        }
    }

    func bench() async {
        guard let llamaContext else {
            return
        }

        messageLog += "\n"
        messageLog += "Running benchmark...\n"
        messageLog += "Model info: "
        messageLog += await llamaContext.model_info() + "\n"

        let t_start = DispatchTime.now().uptimeNanoseconds
        let _ = await llamaContext.bench(pp: 8, tg: 4, pl: 1) // heat up
        let t_end = DispatchTime.now().uptimeNanoseconds

        let t_heat = Double(t_end - t_start) / NS_PER_S
        messageLog += "Heat up time: \(t_heat) seconds, please wait...\n"

        // if more than 5 seconds, then we're probably running on a slow device
        if t_heat > 5.0 {
            messageLog += "Heat up time is too long, aborting benchmark\n"
            return
        }

        let result = await llamaContext.bench(pp: 512, tg: 128, pl: 1, nr: 3)

        messageLog += "\(result)"
        messageLog += "\n"
    }

    func clear() async {
        guard let llamaContext else {
            return
        }

        await llamaContext.clear()
        messageLog = ""
    }

    // MARK: - HTTP Server

    func toggleServer() {
        if serverRunning {
            httpServer.stop()
            serverRunning = false
            serverAddress = ""
            messageLog += "Server stopped\n"
        } else {
            do {
                try httpServer.start(llamaState: self)
                serverRunning = true
                let ip = httpServer.getLocalIP()
                serverAddress = "http://\(ip):8080"
                messageLog += "Server started at \(serverAddress)\n"
            } catch {
                messageLog += "Failed to start server: \(error)\n"
            }
        }
    }

    private func cleanAPIOutput(_ result: String, trim: Bool) -> String {
        var cleaned = result
        // Strip a <think>...</think> block only when it is actually closed. Dropping
        // everything before an *unclosed* <think> (mid-generation, or a reasoning model
        // whose CoT exceeds the token budget) wrongly yields an empty response.
        if let thinkStart = cleaned.range(of: "<think>"),
           let thinkEnd = cleaned.range(of: "</think>", range: thinkStart.upperBound..<cleaned.endIndex) {
            cleaned = String(cleaned[thinkEnd.upperBound...])
        }
        if let endTag = cleaned.range(of: "<|im_end|>") {
            cleaned = String(cleaned[..<endTag.lowerBound])
        }
        return trim ? cleaned.trimmingCharacters(in: .whitespacesAndNewlines) : cleaned
    }

    // Non-streaming completion for API use
    func completeForAPI(messages: [LlamaChatInput], maxTokens: Int = 2048) async -> String {
        guard let llamaContext else { return "Error: No model loaded" }
        guard let prompt = await llamaContext.formatChat(messages: messages, addAssistant: true) else {
            return "Error: Model has no usable embedded chat template"
        }
        await llamaContext.completion_init(text: prompt)
        var result = ""
        var tokenCount = 0
        while await !llamaContext.is_done && tokenCount < maxTokens {
            result += await llamaContext.completion_loop()
            tokenCount += 1
        }
        await llamaContext.clear()
        return cleanAPIOutput(result, trim: true)
    }

    // Streaming completion for OpenAI-compatible SSE clients
    func completeForAPIStreaming(messages: [LlamaChatInput], maxTokens: Int = 2048,
                                 onChunk: @escaping (String) -> Void) async -> String {
        guard let llamaContext else { return "Error: No model loaded" }
        guard let prompt = await llamaContext.formatChat(messages: messages, addAssistant: true) else {
            return "Error: Model has no usable embedded chat template"
        }
        await llamaContext.completion_init(text: prompt)
        var raw = ""
        var emitted = ""
        var tokenCount = 0
        while await !llamaContext.is_done && tokenCount < maxTokens {
            raw += await llamaContext.completion_loop()
            let cleaned = cleanAPIOutput(raw, trim: false)
            if cleaned.hasPrefix(emitted) && cleaned.count > emitted.count {
                let delta = String(cleaned.dropFirst(emitted.count))
                if !delta.isEmpty { onChunk(delta); emitted = cleaned }
            }
            tokenCount += 1
        }
        await llamaContext.clear()
        let final = cleanAPIOutput(raw, trim: false)
        if final.hasPrefix(emitted) && final.count > emitted.count {
            let delta = String(final.dropFirst(emitted.count))
            if !delta.isEmpty { onChunk(delta); emitted = final }
        }
        return final.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
