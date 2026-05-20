# gui frontend for the football analysis pipeline
# wraps the feature modules as subprocesses and shows their output inline

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import subprocess
import threading
import platform
from pathlib import Path
import customtkinter as ctk
from PIL import Image
import config


############################################################
# constants and paths
############################################################

SEQ = config.testSequence
STATIC_CAM_DIR = config.projectRoot / "staticCam"
HOMOGRAPHY_DIR = config.projectRoot / "homographies"
OUTPUT_DIR = config.projectRoot / "output"
OUTPUTS_DIR = config.outputsDir


############################################################
# feature catalog
# each entry maps to a feature module with how to call it
# and where its output lives so the gui can display it
############################################################

FEATURES = [
    {"id": 1, "name": "Player Detection", "module": "detection.py",
     "datasets": ["SportsMOT", "StaticCam"], "argType": "videoInput",
     "output": OUTPUTS_DIR / "detection_test.jpg", "outputType": "image"},

    {"id": 2, "name": "Multi-Object Tracking", "module": "tracking.py",
     "datasets": ["SportsMOT", "StaticCam"], "argType": "videoInput",
     "output": OUTPUTS_DIR / "tracking_test.jpg", "outputType": "image"},

    {"id": 3, "name": "Pitch Homography", "module": "view_homography_video.py",
     "datasets": ["StaticCam"], "argType": "named",
     "output": None, "outputType": "window"},

    {"id": 4, "name": "Optical Flow", "module": "opticalFlow.py",
     "datasets": ["SportsMOT", "StaticCam"], "argType": "videoInput",
     "output": OUTPUTS_DIR / "optical_flow_test.jpg", "outputType": "image"},

    {"id": 5, "name": "Player Movement", "module": "movement.py",
     "datasets": ["SportsMOT", "StaticCam"], "argType": "videoInputHomo",
     "output": OUTPUTS_DIR / "movement_validation.png", "outputType": "image"},

    {"id": 6, "name": "Zone Heatmap", "module": "heatmap.py",
     "datasets": ["StaticCam"], "argType": "positional", "promptUser": True,
     "output": None, "outputType": "window"},

    {"id": 7, "name": "Touch Detection", "module": "touch_pipeline.py",
     "datasets": ["SportsMOT"], "argType": "fixed",
     "fixedArgs": ["--seq", SEQ, "--no-display"],
     "output": config.projectRoot / "touches", "outputType": "folder"},

    {"id": 8, "name": "Team Assignment", "module": "feature_8.py",
     "datasets": ["SportsMOT"], "argType": "fixed",
     "fixedArgs": ["--seq", SEQ],
     "output": OUTPUT_DIR / "feature_8" / SEQ / "overlays", "outputType": "folder"},

    {"id": 9, "name": "Jersey Colour / GK & Ref", "module": "feature_9.py",
     "datasets": ["SportsMOT"], "argType": "fixed",
     "fixedArgs": ["--seq", SEQ],
     "output": OUTPUT_DIR / "feature_9" / SEQ / "output_video.mp4", "outputType": "video"},

    {"id": 10, "name": "Formation Analysis", "module": "feature_10.py",
     "datasets": ["SportsMOT"], "argType": "fixed",
     "fixedArgs": ["--seq", SEQ, "--frames", "300"],
     "output": OUTPUT_DIR / "feature_10" / SEQ / "output_video.mp4", "outputType": "video"},

    {"id": 11, "name": "Ball Detection", "module": "play_ball_detection.py",
     "datasets": ["SportsMOT"], "argType": "interactive",
     "output": None, "outputType": "external"},
]


############################################################
# main application
############################################################

class FootballAnalysisGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.title("Football Analysis Pipeline")
        self.geometry("1300x820")
        self.minsize(1100, 720)

        self.running = False
        self.currentOutput = None
        self.featureButtons = {}

        self.dataset = ctk.StringVar(value="SportsMOT")
        self.clipVar = ctk.StringVar()
        self.homoVar = ctk.StringVar()

        self._scanStaticFiles()
        self._buildUI()
        self._onDatasetChange()

    ############################################################
    # setup
    ############################################################

    def _scanStaticFiles(self):
        if STATIC_CAM_DIR.exists():
            self.clipFiles = sorted([p.name for p in STATIC_CAM_DIR.glob("*.mp4")])

        else:
            self.clipFiles = []

        if HOMOGRAPHY_DIR.exists():
            self.homoFiles = sorted([p.name for p in HOMOGRAPHY_DIR.glob("*.npz")])

        else:
            self.homoFiles = []

    def _buildUI(self):
        # ── sidebar ────────────────────────────────────────────
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(self.sidebar, text="⚽  Football Analysis",
                     font=ctk.CTkFont(size=17, weight="bold")
        ).pack(pady=(20, 22), padx=20, anchor="w")

        # dataset section
        ctk.CTkLabel(self.sidebar, text="DATASET",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#9ca3af"
        ).pack(anchor="w", padx=20, pady=(0, 5))

        for ds in ["SportsMOT", "StaticCam"]:
            rb = ctk.CTkRadioButton(self.sidebar, text=ds,
                                    variable=self.dataset, value=ds,
                                    command=self._onDatasetChange)
            rb.pack(anchor="w", padx=25, pady=3)

        # clip selection (packed only when StaticCam is active)
        self.clipFrame = ctk.CTkFrame(self.sidebar, fg_color="transparent")

        ctk.CTkLabel(self.clipFrame, text="CLIP",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#9ca3af"
        ).pack(anchor="w", pady=(10, 3))

        clipChoices = self.clipFiles if self.clipFiles else ["(no clips found)"]
        self.clipMenu = ctk.CTkOptionMenu(self.clipFrame, variable=self.clipVar,
                                          values=clipChoices)
        self.clipMenu.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(self.clipFrame, text="HOMOGRAPHY",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#9ca3af"
        ).pack(anchor="w", pady=(0, 3))

        homoChoices = self.homoFiles if self.homoFiles else ["(no homographies found)"]
        self.homoMenu = ctk.CTkOptionMenu(self.clipFrame, variable=self.homoVar,
                                          values=homoChoices)
        self.homoMenu.pack(fill="x", pady=(0, 5))

        # sensible defaults pointing at the validated combo
        if "tactical_clipC.mp4" in self.clipFiles:
            self.clipVar.set("tactical_clipC.mp4")

        elif self.clipFiles:
            self.clipVar.set(self.clipFiles[0])

        if "napoli_roma_2.npz" in self.homoFiles:
            self.homoVar.set("napoli_roma_2.npz")

        elif self.homoFiles:
            self.homoVar.set(self.homoFiles[0])

        # features section header acts as the anchor for clipFrame
        self.featuresHeader = ctk.CTkLabel(self.sidebar, text="FEATURES",
                                           font=ctk.CTkFont(size=11, weight="bold"),
                                           text_color="#9ca3af")
        self.featuresHeader.pack(anchor="w", padx=20, pady=(15, 5))

        for feat in FEATURES:
            btn = ctk.CTkButton(self.sidebar,
                                text=f"  F{feat['id']}    {feat['name']}",
                                anchor="w",
                                height=32,
                                corner_radius=6,
                                command=lambda f=feat: self._runFeature(f))
            btn.pack(fill="x", padx=15, pady=2)
            self.featureButtons[feat["id"]] = btn

        # ── main content area ──────────────────────────────────
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(side="right", fill="both", expand=True, padx=14, pady=14)

        # image display frame
        self.imageFrame = ctk.CTkFrame(self.content, fg_color="#0d1117")
        self.imageFrame.pack(fill="both", expand=True, pady=(0, 8))

        self.imageLabel = ctk.CTkLabel(self.imageFrame,
                                       text="run a feature to see output here",
                                       text_color="#6b7280",
                                       font=ctk.CTkFont(size=13))
        self.imageLabel.pack(expand=True, padx=10, pady=10)

        # open output button (shown only when a result is available)
        self.openOutputBtn = ctk.CTkButton(self.content, text="Open Output",
                                           command=self._openOutput,
                                           width=140, height=32)

        # terminal label
        ctk.CTkLabel(self.content, text="TERMINAL",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#9ca3af"
        ).pack(anchor="w", pady=(2, 3))

        # terminal textbox
        self.terminal = ctk.CTkTextbox(self.content, height=170,
                                       fg_color="#0d1117",
                                       text_color="#d1fae5",
                                       font=ctk.CTkFont(family="Consolas", size=11))
        self.terminal.pack(fill="x")
        self.terminal.configure(state="disabled")

        # status bar
        self.statusLabel = ctk.CTkLabel(self.content, text="ready",
                                        text_color="#9ca3af",
                                        font=ctk.CTkFont(size=11))
        self.statusLabel.pack(anchor="w", pady=(5, 0))

    ############################################################
    # ui state
    ############################################################

    def _onDatasetChange(self):
        ds = self.dataset.get()
        if(ds == "StaticCam"):
            self.clipFrame.pack(fill="x", padx=20, before=self.featuresHeader)

        else:
            self.clipFrame.pack_forget()

        self._updateFeatureButtons()

    def _updateFeatureButtons(self):
        ds = self.dataset.get()
        for feat in FEATURES:
            btn = self.featureButtons[feat["id"]]
            if(ds in feat["datasets"]):
                btn.configure(state="normal")

            else:
                btn.configure(state="disabled")

    def _log(self, text):
        self.terminal.configure(state="normal")
        self.terminal.insert("end", text)
        self.terminal.see("end")
        self.terminal.configure(state="disabled")

    def _clearTerminal(self):
        self.terminal.configure(state="normal")
        self.terminal.delete("1.0", "end")
        self.terminal.configure(state="disabled")

    def _resetImageLabel(self, text, textColor="#9ca3af"):
        # ctklabel crashes if you set image=None on a label that already had
        # a ctkimage, so we destroy and recreate it instead
        self.imageLabel.destroy()
        self.imageLabel = ctk.CTkLabel(self.imageFrame,
                                       text=text,
                                       text_color=textColor,
                                       font=ctk.CTkFont(size=13))
        self.imageLabel.pack(expand=True, padx=10, pady=10)

    ############################################################
    # feature execution
    ############################################################

    def _buildCommand(self, feat):
        cmd = [sys.executable, feat["module"]]
        argType = feat["argType"]
        ds = self.dataset.get()

        if(argType == "fixed"):
            cmd += feat["fixedArgs"]

        elif(argType == "videoInput"):
            # F1, F2, F4: pass --input when StaticCam, nothing for SportsMOT (uses config default)
            if(ds == "StaticCam"):
                clip = self.clipVar.get()
                cmd += ["--input", f"staticCam/{clip}"]

        elif(argType == "videoInputHomo"):
            # F5: pass --input and --homo when StaticCam
            if(ds == "StaticCam"):
                clip = self.clipVar.get()
                homo = self.homoVar.get()
                cmd += ["--input", f"staticCam/{clip}",
                        "--homo", f"homographies/{homo}"]

        elif(argType == "named"):
            clip = self.clipVar.get()
            homo = self.homoVar.get()
            cmd += ["--video", f"staticCam/{clip}",
                    "--homo", f"homographies/{homo}"]

        elif(argType == "positional"):
            clip = self.clipVar.get()
            homo = self.homoVar.get()
            cmd += [f"staticCam/{clip}", f"homographies/{homo}"]

        return cmd

    def _runFeature(self, feat):
        if self.running:
            return

        # interactive features open in a separate terminal window
        if(feat["argType"] == "interactive"):
            self._launchExternal(feat)
            return

        # features that need user input mid-run get launched with args but in a new terminal
        if(feat.get("promptUser")):
            cmd = self._buildCommand(feat)
            self._launchExternalWithCmd(feat, cmd)
            return

        # check clip selection is valid for features that need it
        if(feat["argType"] in ("named", "positional", "videoInput", "videoInputHomo")):
            if(self.dataset.get() == "StaticCam"):
                clip = self.clipVar.get()
                homo = self.homoVar.get()
                if(not clip or not homo or "(no" in clip or "(no" in homo):
                    self._clearTerminal()
                    self._log("error: clip or homography not selected\n")
                    self.statusLabel.configure(text="missing clip or homography",
                                               text_color="#f87171")
                    return

        # reset the right-hand panel
        self._clearTerminal()
        self.currentOutput = None
        self.openOutputBtn.pack_forget()
        self._resetImageLabel("running...")

        self.running = True
        self.statusLabel.configure(text=f"running F{feat['id']}", text_color="#facc15")

        for btn in self.featureButtons.values():
            btn.configure(state="disabled")

        cmd = self._buildCommand(feat)
        thread = threading.Thread(target=self._runProcess,
                                  args=(feat, cmd),
                                  daemon=True)
        thread.start()

    def _runProcess(self, feat, cmd):
        cmdStr = " ".join(str(c) for c in cmd)
        self.after(0, self._log, f"running: {cmdStr}\n\n")

        try:
            proc = subprocess.Popen(cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                    bufsize=1,
                                    cwd=str(config.projectRoot))
            for line in proc.stdout:
                self.after(0, self._log, line)

            proc.wait()
            returnCode = proc.returncode

        except Exception as e:
            self.after(0, self._log, f"\nerror running subprocess: {e}\n")
            returnCode = -1

        self.after(0, self._onProcessDone, feat, returnCode)

    def _onProcessDone(self, feat, returnCode):
        self.running = False
        self._updateFeatureButtons()

        if(returnCode == 0):
            self.statusLabel.configure(text=f"F{feat['id']} complete",
                                       text_color="#4ade80")

        else:
            self.statusLabel.configure(text=f"F{feat['id']} exited with code {returnCode}",
                                       text_color="#f87171")

        self._showOutput(feat)

    def _showOutput(self, feat):
        outPath = feat.get("output")
        outType = feat.get("outputType", "none")

        if(outPath is None):
            self.imageLabel.configure(text=f"F{feat['id']} produced no file output",
                                       text_color="#9ca3af")
            return

        outPath = Path(outPath)

        if(outType == "image"):
            if outPath.exists():
                self._setImage(outPath)
                self.currentOutput = outPath
                self.openOutputBtn.configure(text="Open Image")
                self.openOutputBtn.pack(anchor="w", pady=(0, 8), before=self.terminal)

            else:
                self.imageLabel.configure(text=f"expected output not found at {outPath.name}",
                                           text_color="#f87171")

        elif(outType == "video"):
            if outPath.exists():
                self.imageLabel.configure(
                    text=f"video ready  -  {outPath.name}\nclick the button below to play",
                    text_color="#4ade80"
                )
                self.currentOutput = outPath
                self.openOutputBtn.configure(text="Play Video")
                self.openOutputBtn.pack(anchor="w", pady=(0, 8), before=self.terminal)

            else:
                self.imageLabel.configure(text=f"expected video not found at {outPath.name}",
                                           text_color="#f87171")

        elif(outType == "folder"):
            if outPath.exists():
                images = sorted(list(outPath.glob("*.jpg")) + list(outPath.glob("*.png")))
                if images:
                    self._setImage(images[0])

                else:
                    self.imageLabel.configure(text=f"output folder ready  -  {outPath.name}",
                                               text_color="#4ade80")

                self.currentOutput = outPath
                self.openOutputBtn.configure(text="Open Folder")
                self.openOutputBtn.pack(anchor="w", pady=(0, 8), before=self.terminal)

            else:
                self.imageLabel.configure(text=f"expected folder not found at {outPath.name}",
                                           text_color="#f87171")

        elif(outType == "window"):
            self.imageLabel.configure(text=f"F{feat['id']} ran in an external window",
                                       text_color="#9ca3af")

    ############################################################
    # output handling
    ############################################################

    def _setImage(self, path):
        try:
            self.update_idletasks()
            pil = Image.open(path)
            frameW = max(self.imageFrame.winfo_width() - 30, 500)
            frameH = max(self.imageFrame.winfo_height() - 30, 280)
            scale = min(frameW / pil.width, frameH / pil.height, 1.0)
            newW = max(int(pil.width * scale), 1)
            newH = max(int(pil.height * scale), 1)
            ctkImg = ctk.CTkImage(light_image=pil, dark_image=pil, size=(newW, newH))
            self._currentCtkImg = ctkImg  # keep reference to prevent gc
            self.imageLabel.configure(image=ctkImg, text="")

        except Exception as e:
            self._resetImageLabel(f"could not load image: {e}", "#f87171")

    def _launchExternal(self, feat):
        module = feat["module"]
        self._log(f"launching {module} in a new terminal\n")
        try:
            if(platform.system() == "Windows"):
                subprocess.Popen(
                    f'start "F{feat["id"]} - {feat["name"]}" cmd /k "{sys.executable}" {module}',
                    shell=True,
                    cwd=str(config.projectRoot),
                )

            elif(platform.system() == "Darwin"):
                script = (f'tell application "Terminal" to do script '
                          f'"cd {config.projectRoot} && {sys.executable} {module}"')
                subprocess.Popen(["osascript", "-e", script])

            else:
                subprocess.Popen(["xterm", "-e", sys.executable, module],
                                 cwd=str(config.projectRoot))

            self.statusLabel.configure(text=f"F{feat['id']} launched in new terminal",
                                       text_color="#4ade80")

        except Exception as e:
            self._log(f"could not launch terminal: {e}\n")
            self.statusLabel.configure(text="terminal launch failed", text_color="#f87171")

    def _launchExternalWithCmd(self, feat, cmd):
        """like _launchExternal but uses a pre-built command list (for features that need args)"""
        # quote each part that has spaces so the shell receives it correctly
        parts = []
        for c in cmd:
            s = str(c)
            if(" " in s):
                parts.append(f'"{s}"')
            else:
                parts.append(s)
        cmdStr = " ".join(parts)
        self._log(f"launching F{feat['id']} in a new terminal\ncommand: {cmdStr}\n")
        try:
            if(platform.system() == "Windows"):
                subprocess.Popen(
                    f'start "F{feat["id"]} - {feat["name"]}" cmd /k {cmdStr}',
                    shell=True,
                    cwd=str(config.projectRoot),
                )

            elif(platform.system() == "Darwin"):
                script = (f'tell application "Terminal" to do script '
                          f'"cd {config.projectRoot} && {cmdStr}"')
                subprocess.Popen(["osascript", "-e", script])

            else:
                subprocess.Popen(["xterm", "-e"] + cmd,
                                 cwd=str(config.projectRoot))

            self.statusLabel.configure(text=f"F{feat['id']} launched in new terminal",
                                       text_color="#4ade80")
            self.statusLabel.configure(text=f"F{feat['id']} launched in new terminal",
                                       text_color="#4ade80")

            self._resetImageLabel("heatmap running in external terminal\nclick the button below once it is done")
            self.openOutputBtn.configure(text="Load Latest Output",
                                         command=self._loadLatestOutput)
            self.openOutputBtn.pack(anchor="w", pady=(0, 8), before=self.terminal)

        except Exception as e:
            self._log(f"could not launch terminal: {e}\n")
            self.statusLabel.configure(text="terminal launch failed", text_color="#f87171")
    
    def _loadLatestOutput(self):
        pngs = list(OUTPUTS_DIR.glob("*.png"))
        if(not pngs):
            self._log("no png files found in outputs/\n")
            return
        latest = max(pngs, key=lambda p: p.stat().st_mtime)
        self._resetImageLabel("")
        self._setImage(latest)
        self.currentOutput = latest
        self.openOutputBtn.configure(text="Open Image",
                                    command=self._openOutput)
        self.statusLabel.configure(text=f"loaded {latest.name}", text_color="#4ade80")

    def _openOutput(self):
        if self.currentOutput is None:
            return

        path = Path(self.currentOutput)
        if not path.exists():
            self._log(f"output not found at {path}\n")
            return

        try:
            if(platform.system() == "Windows"):
                os.startfile(str(path))

            elif(platform.system() == "Darwin"):
                subprocess.Popen(["open", str(path)])

            else:
                subprocess.Popen(["xdg-open", str(path)])

        except Exception as e:
            self._log(f"could not open output: {e}\n")


############################################################
# entry point
############################################################

if __name__ == "__main__":
    app = FootballAnalysisGUI()
    app.mainloop()