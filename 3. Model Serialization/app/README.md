# Run the app with a live HTTP server

You must start a live HTTP server **inside this folder** (`3. Model Serialization/app`) before opening the app.

## Why
The app loads local assets (`app.js`, `style.css`) and can fail with browser security restrictions if opened directly with `file://`.

## Option 1: VS Code Live Server
1. Open this `app` folder in VS Code.
2. Right-click `index.html`.
3. Click **Open with Live Server**.

## Option 2: Python HTTP server
From a terminal opened in this folder, run:

```bash
python -m http.server 5500
```

Then open:
- http://localhost:5500/index.html

## Important
If you run the server from a different directory, paths may break. Always start it in this `app` folder.
