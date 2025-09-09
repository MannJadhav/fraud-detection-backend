// background.js
chrome.runtime.onInstalled.addListener(() => {
    console.log("Realtime Fraud Detector extension installed.");
});

// Optional: listen for messages from popup or content scripts
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    console.log("Background received:", msg);
    sendResponse({ status: "received" });
});
