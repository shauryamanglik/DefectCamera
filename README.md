# DefectCam — setup & test guide

On-device defect detector. Runs in the browser, uses your Mac or iPhone camera, no cloud, no cost, no limits. First load needs internet (to fetch the model, ~5MB); after that it works offline.

## Fastest test (MacBook, 30 seconds)

1. Unzip the folder somewhere.
2. Open Terminal, `cd` into the folder, run:
   ```
   python3 -m http.server 8000
   ```
3. Open Chrome or Safari to `http://localhost:8000`
4. Camera works on localhost even without HTTPS. Wait for status to read "model ready."

## Use it on your iPhone

The iPhone camera needs **HTTPS** (localhost is exempt, your phone is not). Two easy ways:

**Option A — ngrok (quickest):**
```
python3 -m http.server 8000
# in a second terminal:
npx ngrok http 8000
```
Open the `https://…ngrok…` URL it prints, on your iPhone in Safari. Allow camera.

**Option B — deploy free:** drag the folder onto https://app.netlify.com/drop . You get a permanent HTTPS link that works on any phone.

Once open on the iPhone in Safari: tap Share → **Add to Home Screen**. It now launches like a real app, fullscreen.

## How to train and test

**Train tab:**
1. Start camera.
2. **Capture GOOD:** hold a known-good part, tap ✓ Capture GOOD. Do this 15–30 times, varied angles and lighting. This defines "normal."
3. **Capture BAD:** hold a defective part, tap ✕ Capture BAD. The paint screen opens. Paint directly over the exact defect (scratch/discoloration/etc), pick the label, tap Save. Repeat for each bad part.
4. Tap **Build detector from samples.**

**Inspect tab:**
1. Start camera → Start live inspection.
2. Green **PASS** = looks normal. Red **DEFECT** = deviation found, with red boxes over the exact bad regions and the label if it matches a trained defect type.
3. **Sensitivity slider:** lower = stricter. If good parts show red, raise it. If real defects slip through, lower it.

**Data tab:** review/delete samples, and **Export dataset** to a file so you never lose your training (localStorage can get cleared). Load it back anytime.

## Reality check
- Strong on obvious stuff: dents, clear discoloration, coating gaps, missing coating.
- Subtle fine scratches / faint rainbowing need more examples (30+ of that type) and good lighting, and may still be at the edge of what a phone cam + lightweight model resolves. That's the free/on-device tradeoff vs an IV4/CV-X.
- Consistent lighting during training AND inspection matters more than anything else. Shadows read as defects.
