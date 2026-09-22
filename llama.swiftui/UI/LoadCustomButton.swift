import SwiftUI
import UniformTypeIdentifiers

struct LoadCustomButton: View {
    @ObservedObject private var llamaState: LlamaState
    @State private var showFileImporter = false

    init(llamaState: LlamaState) {
        self.llamaState = llamaState
    }

    var body: some View {
        VStack {
            Button(action: {
                showFileImporter = true
            }) {
                Text("Load Custom Model")
            }
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [UTType(filenameExtension: "gguf", conformingTo: .data)!],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let files):
                files.forEach { file in
                    let gotAccess = file.startAccessingSecurityScopedResource()
                    if !gotAccess { return }

                    // Copy the picked .gguf into Documents so it persists and the API
                    // can re-load it by filename later; then load it.
                    let dest = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                        .appendingPathComponent(file.lastPathComponent)
                    if !FileManager.default.fileExists(atPath: dest.path) {
                        try? FileManager.default.copyItem(at: file, to: dest)
                    }
                    file.stopAccessingSecurityScopedResource()
                    llamaState.loadModel(modelUrl: dest)
                }
            case .failure(let error):
                print(error)
            }
        }
    }
}
