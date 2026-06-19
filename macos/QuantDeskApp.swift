import Cocoa
import WebKit

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var statusLabel: NSTextField!
    private var retryButton: NSButton!
    private var serverProcess: Process?
    private var startedOwnServer = false

    private let appName = "QuantDesk"
    private let appURL = URL(string: "http://localhost:8888/")!
    private let healthURL = URL(string: "http://localhost:8888/api/stocks")!

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        buildWindow()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        bootServerAndLoad()
    }

    func applicationWillTerminate(_ notification: Notification) {
        if startedOwnServer {
            serverProcess?.terminate()
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    private func buildWindow() {
        let config = WKWebViewConfiguration()
        config.preferences.javaScriptCanOpenWindowsAutomatically = true

        webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = self
        webView.allowsBackForwardNavigationGestures = true

        statusLabel = NSTextField(labelWithString: "正在启动 QuantDesk 后端...")
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.font = .systemFont(ofSize: 13, weight: .medium)
        statusLabel.alignment = .center

        retryButton = NSButton(title: "重新连接", target: self, action: #selector(retry))
        retryButton.bezelStyle = .rounded
        retryButton.isHidden = true

        let overlay = NSStackView(views: [statusLabel, retryButton])
        overlay.orientation = .vertical
        overlay.alignment = .centerX
        overlay.spacing = 14
        overlay.translatesAutoresizingMaskIntoConstraints = false

        let container = NSView()
        container.wantsLayer = true
        container.layer?.backgroundColor = NSColor(red: 0.027, green: 0.039, blue: 0.059, alpha: 1).cgColor
        container.addSubview(webView)
        container.addSubview(overlay)
        webView.translatesAutoresizingMaskIntoConstraints = false

        NSLayoutConstraint.activate([
            webView.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            webView.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            webView.topAnchor.constraint(equalTo: container.topAnchor),
            webView.bottomAnchor.constraint(equalTo: container.bottomAnchor),
            overlay.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            overlay.centerYAnchor.constraint(equalTo: container.centerYAnchor)
        ])

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1440, height: 920),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = appName
        window.minSize = NSSize(width: 1080, height: 720)
        window.center()
        window.contentView = container
        window.titlebarAppearsTransparent = true
        window.toolbarStyle = .unifiedCompact
    }

    @objc private func retry() {
        retryButton.isHidden = true
        statusLabel.stringValue = "正在重新连接 QuantDesk..."
        bootServerAndLoad()
    }

    private func bootServerAndLoad() {
        Task {
            if await waitForHealth(seconds: 1) {
                await loadApp()
                return
            }

            do {
                try startServer()
            } catch {
                await showError("启动后端失败：\(error.localizedDescription)")
                return
            }

            if await waitForHealth(seconds: 45) {
                await loadApp()
            } else {
                await showError("后端启动超时。请查看 ~/Library/Application Support/QuantDesk/logs/server.log")
            }
        }
    }

    private func startServer() throws {
        let resourceURL = Bundle.main.resourceURL!
        let appDir = resourceURL.appendingPathComponent("app")
        let supportDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/QuantDesk", isDirectory: true)
        let logDir = supportDir.appendingPathComponent("logs", isDirectory: true)
        try FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)

        guard let python = findPythonWithDependencies() else {
            throw NSError(
                domain: "QuantDesk",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "没有找到带 FastAPI 依赖的 Python。请安装 fastapi、uvicorn、python-docx、requests。"]
            )
        }

        let logURL = logDir.appendingPathComponent("server.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let logHandle = try FileHandle(forWritingTo: logURL)
        logHandle.write("Using Python: \(python)\n".data(using: .utf8)!)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = ["server.py"]
        process.currentDirectoryURL = appDir
        var environment = [
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONUNBUFFERED": "1",
            "HOME": FileManager.default.homeDirectoryForCurrentUser.path
        ]
        if let key = ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"], !key.isEmpty {
            environment["ANTHROPIC_API_KEY"] = key
        }
        process.environment = environment
        process.standardOutput = logHandle
        process.standardError = logHandle
        try process.run()

        serverProcess = process
        startedOwnServer = true
    }

    private func findPythonWithDependencies() -> String? {
        let candidates = [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3"
        ]

        for candidate in candidates {
            guard FileManager.default.isExecutableFile(atPath: candidate) else {
                continue
            }
            let process = Process()
            process.executableURL = URL(fileURLWithPath: candidate)
            process.arguments = ["-c", "import fastapi, uvicorn"]
            process.standardOutput = Pipe()
            process.standardError = Pipe()
            do {
                try process.run()
                process.waitUntilExit()
                if process.terminationStatus == 0 {
                    return candidate
                }
            } catch {
                continue
            }
        }

        return nil
    }

    private func waitForHealth(seconds: Int) async -> Bool {
        let attempts = max(1, seconds * 2)
        for _ in 0..<attempts {
            if await isHealthy() {
                return true
            }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        return false
    }

    private func isHealthy() async -> Bool {
        var request = URLRequest(url: healthURL)
        request.timeoutInterval = 2
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    @MainActor private func loadApp() {
        statusLabel.isHidden = true
        retryButton.isHidden = true
        webView.load(URLRequest(url: appURL))
    }

    @MainActor private func showError(_ message: String) {
        statusLabel.isHidden = false
        retryButton.isHidden = false
        statusLabel.stringValue = message
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        showWebError(error)
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        showWebError(error)
    }

    private func showWebError(_ error: Error) {
        statusLabel.isHidden = false
        retryButton.isHidden = false
        statusLabel.stringValue = "页面加载失败：\(error.localizedDescription)"
    }
}

@main
struct QuantDeskMain {
    private static var delegate: AppDelegate?

    static func main() {
        let app = NSApplication.shared
        let appDelegate = AppDelegate()
        delegate = appDelegate
        app.delegate = appDelegate
        app.run()
    }
}
