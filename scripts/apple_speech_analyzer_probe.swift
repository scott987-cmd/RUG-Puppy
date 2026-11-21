// Isolated, file-only comparison probe. It never opens an input device or plays audio.
import AVFoundation
import Foundation
import Speech

@main
struct AppleSpeechAnalyzerProbe {
    static func main() async throws {
        var arguments = Array(CommandLine.arguments.dropFirst())
        if arguments.first == "--" {
            arguments.removeFirst()
        }
        let installingEnglish = arguments == ["--install-en"]
        guard installingEnglish || (1...2).contains(arguments.count) else {
            fputs("Usage: apple_speech_analyzer_probe <audio-file> [en_US] | --install-en\n", stderr)
            exit(2)
        }

        let localeName = installingEnglish || arguments.count == 2 ? "en_US" : "zh_CN"
        if arguments.count == 2 && arguments[1] != "en_US" {
            throw ProbeError.localeUnavailable
        }
        guard SpeechTranscriber.isAvailable else {
            throw ProbeError.unavailable
        }
        let locale = Locale(identifier: localeName)
        let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
        if installingEnglish {
            _ = try await AssetInventory.reserve(locale: locale)
            if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                print("Installing local English speech recognition", terminator: "\n")
                try await request.downloadAndInstall()
            }
            print("English speech recognition ready")
            return
        }
        let installed = await SpeechTranscriber.installedLocales
        guard installed.contains(where: { $0.identifier == locale.identifier }) else {
            throw ProbeError.localeUnavailable
        }

        let analyzer = SpeechAnalyzer(modules: [transcriber])
        if localeName == "en_US" {
            // Hint the whole beginner vocabulary, never only the expected word.
            let context = AnalysisContext()
            context.contextualStrings[.general] = ["dog", "cat", "ball", "apple", "red", "blue", "hello", "bye", "hand", "eye"]
            try await analyzer.setContext(context)
        }
        let audioFile = try AVAudioFile(forReading: URL(fileURLWithPath: arguments[0]))
        let resultTask = Task { () throws -> String in
            var text = ""
            for try await result in transcriber.results {
                if result.isFinal {
                    text += String(result.text.characters)
                }
            }
            return text
        }

        if let end = try await analyzer.analyzeSequence(from: audioFile) {
            try await analyzer.finalizeAndFinish(through: end)
        } else {
            await analyzer.cancelAndFinishNow()
        }
        print(try await resultTask.value)
    }
}

enum ProbeError: Error {
    case unavailable
    case localeUnavailable
}
