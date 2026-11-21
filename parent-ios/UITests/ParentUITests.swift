import XCTest

final class ParentUITests: XCTestCase {
    private func tapTab(_ label: String, in app: XCUIApplication) {
        // iPhone uses a TabBar; iPadOS exposes floating tabs directly as
        // Buttons under the application instead of descendants of TabBar.
        let tab = app.buttons.matching(NSPredicate(format: "label == %@", label)).firstMatch
        XCTAssertTrue(tab.waitForExistence(timeout: 5), "Missing tab: \(label)")
        tab.tap()
    }

    func testPublicDemoCannotDriveRobot() {
        let app = XCUIApplication()
        app.launch()
        tapTab("连接", in: app)
        let enterDemo = app.buttons["enterDemoButton"]
        XCTAssertTrue(enterDemo.waitForExistence(timeout: 10))
        enterDemo.tap()
        XCTAssertTrue(app.staticTexts["demoBanner"].waitForExistence(timeout: 10))
        XCTAssertFalse(app.buttons["photoButton"].isEnabled)
        XCTAssertFalse(app.buttons["stopButton"].isEnabled)
        XCTAssertFalse(app.buttons["patrolButton"].isEnabled)
        XCTAssertTrue(app.buttons["exitDemoButton"].exists)
        app.buttons["exitDemoButton"].tap()
        XCTAssertFalse(app.staticTexts["demoBanner"].exists)
    }

    func testFamilyEnrollmentRequiresTrustedHomeConnection() {
        let app = XCUIApplication()
        app.launchArguments = ["--ui-demo"]
        app.launch()
        let enrollmentShortcut = app.buttons["openVoiceEnrollmentButton"]
        XCTAssertTrue(enrollmentShortcut.waitForExistence(timeout: 10))
        enrollmentShortcut.tap()
        XCTAssertTrue(app.staticTexts["等待家庭连接"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["startVoiceEnrollmentButton"].exists)
    }

    func testDemoStatusRequestAndPairingReview() {
        let app = XCUIApplication()
        app.launchArguments = ["--ui-demo"]
        app.launch()
        XCTAssertTrue(app.staticTexts["demoBanner"].waitForExistence(timeout: 10))
        let statusButton = app.buttons["statusButton"]
        XCTAssertEqual(XCTWaiter.wait(for: [XCTNSPredicateExpectation(predicate: NSPredicate(format: "enabled == true"), object: statusButton)], timeout: 20), .completed)
        XCTAssertFalse(app.buttons["photoButton"].isEnabled)
        XCTAssertFalse(app.buttons["stopButton"].isEnabled)
        statusButton.tap()
        let feedback = app.staticTexts["homeCommandFeedback"]
        XCTAssertTrue(feedback.waitForExistence(timeout: 20))
        XCTAssertTrue(feedback.label.contains("已提交"), "首页应说明请求已提交，不能暗示小狗已执行")
        XCTAssertTrue(statusButton.label.contains("查看状态"), "发送结束后按钮应恢复原样")
        let refreshButton = app.buttons["homeRefreshButton"]
        XCTAssertEqual(XCTWaiter.wait(for: [XCTNSPredicateExpectation(predicate: NSPredicate(format: "enabled == true"), object: refreshButton)], timeout: 20), .completed)
        refreshButton.tap()
        XCTAssertTrue(app.staticTexts["已检查家中状态"].waitForExistence(timeout: 20))
        XCTAssertTrue(refreshButton.exists, "刷新完成后按钮应恢复")
        tapTab("记录", in: app)
        let submitted = app.staticTexts["已提交"].waitForExistence(timeout: 20)
        if !submitted { print("Demo submission UI: \(app.debugDescription)") }
        XCTAssertTrue(submitted)
        tapTab("连接", in: app)
        XCTAssertTrue(app.buttons["importPairingButton"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["服务器, demo-home-mac"].exists)
        app.swipeUp()
        XCTAssertTrue(app.buttons["copyPublicKeyButton"].exists)
    }

    func testStaleSnapshotIsNotOnlineAndVerifiedPhotoIsVisible() {
        let app = XCUIApplication()
        app.launchArguments = ["--ui-demo", "--stale", "--no-description"]
        app.launch()
        XCTAssertTrue(app.staticTexts["状态已过期"].waitForExistence(timeout: 25))
        XCTAssertFalse(app.staticTexts["在线"].exists)
        app.swipeUp()
        XCTAssertTrue(app.images["verifiedPhoto"].waitForExistence(timeout: 20))
        XCTAssertTrue(app.staticTexts["已验证服务器签名和照片完整性"].exists)
    }

    func testPhotoSummaryAndUnavailableSummaryKeepPhotoVisible() {
        let app = XCUIApplication()
        app.launchArguments = ["--ui-demo"]
        app.launch()
        XCTAssertTrue(app.staticTexts["demoBanner"].waitForExistence(timeout: 10))
        tapTab("记录", in: app)
        XCTAssertTrue(app.staticTexts["photoDescriptionReply"].waitForExistence(timeout: 5))
        XCTAssertEqual(app.staticTexts["photoDescriptionReply"].label, "照片里能看到浅绿色背景和爪印，无法判断房间其他位置。")
        XCTAssertTrue(app.staticTexts["photoDescriptionScope"].exists)
        XCTAssertTrue(app.staticTexts["画面信息有限，描述可能不完整。"].exists)
        app.terminate()
        app.launchArguments = ["--ui-demo", "--description-unavailable"]
        app.launch()
        XCTAssertTrue(app.staticTexts["demoBanner"].waitForExistence(timeout: 10))
        app.swipeUp()
        XCTAssertTrue(app.images["verifiedPhoto"].waitForExistence(timeout: 5))
        app.swipeUp()
        XCTAssertTrue(app.staticTexts["photoDescriptionUnavailable"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.staticTexts["photoDescriptionReply"].exists)
    }

    func testMissingRemotePhotoIsNotPresentedAsVerified() {
        let app = XCUIApplication()
        app.launchArguments = ["--ui-demo", "--missing-photo", "--description-unavailable"]
        app.launch()
        XCTAssertTrue(app.staticTexts["demoBanner"].waitForExistence(timeout: 10))
        tapTab("记录", in: app)
        XCTAssertTrue(app.staticTexts["photoUnavailable"].waitForExistence(timeout: 20))
        XCTAssertTrue(app.staticTexts["photoDescriptionUnavailable"].waitForExistence(timeout: 20))
        tapTab("小狗", in: app)
        XCTAssertFalse(app.images["verifiedPhoto"].exists)
    }

    func testTrustedMobileAdminConnectionOnConfiguredPhone() throws {
        guard ProcessInfo.processInfo.environment["PUPPY_REAL_ADMIN_TEST"] == "1" else {
            throw XCTSkip("仅在已配对的家庭真机上运行")
        }
        let app = XCUIApplication()
        app.launch()
        tapTab("连接", in: app)
        let signed = app.staticTexts["已由家庭服务器签名"]
        if !signed.waitForExistence(timeout: 40) { print("Connection UI: \(app.debugDescription)") }
        XCTAssertTrue(signed.exists)
        let check = app.buttons["检查家中管理连接"]
        XCTAssertTrue(check.waitForExistence(timeout: 5))
        check.tap()
        let success = app.staticTexts["已通过手机签名连接家中 Mac"]
        if !success.waitForExistence(timeout: 25) { print("Admin result UI: \(app.debugDescription)") }
        XCTAssertTrue(success.exists)
    }
}
