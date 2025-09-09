// content_script.js
// Example: if the page has inputs like <input data-tx-amt="..."> etc.
(function () {
    try {
        const features = {};
        ['amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long'].forEach(k => {
            const el = document.querySelector(`[data-tx-${k}]`);
            if (el) {
                let val = el.getAttribute(`data-tx-${k}`) || el.value || el.innerText;
                if (!isNaN(val)) val = Number(val);
                features[k] = val;
            }
        });
        if (Object.keys(features).length) {
            // store to chrome storage for popup to read
            chrome.storage.local.set({ autoFeatures: JSON.stringify(features) });
        }
    } catch (e) {
        console.warn('content script error', e);
    }
})();
