// Ties each video to its score: a playhead at the current time, the layers on screen lit, click to seek.
// Also play/pause and sound buttons, and no autoplay for people who ask for reduced motion.
(function () {
  "use strict";
  var root = document.documentElement;
  root.classList.add("js");
  var still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  Array.prototype.forEach.call(document.querySelectorAll(".player"), function (fig) {
    var v = fig.querySelector("video");
    var score = fig.querySelector(".score");
    var play = fig.querySelector("[data-play]");
    var sound = fig.querySelector("[data-sound]");
    var tc = fig.querySelector(".tc");
    if (!v) return;
    v.removeAttribute("controls");  // the buttons below replace them; without this script the native ones stay
    if (still) {
      v.removeAttribute("autoplay");
      v.pause();
    }
    var dur = score ? parseFloat(score.getAttribute("data-dur")) : 0;
    var bars = score ? Array.prototype.map.call(score.querySelectorAll(".bar"), function (b) {
      return [b, parseFloat(b.getAttribute("data-in")), parseFloat(b.getAttribute("data-out"))];
    }) : [];
    if (score) score.setAttribute("data-live", "");
    var raf = 0;

    function paint() {
      var t = v.currentTime || 0;
      if (score && dur) score.style.setProperty("--t", Math.min(t / dur, 1).toFixed(4));
      for (var i = 0; i < bars.length; i++) bars[i][0].classList.toggle("on", t >= bars[i][1] && t < bars[i][2]);
      if (tc) tc.textContent = t.toFixed(1) + " s / " + (dur || v.duration || 0).toFixed(1) + " s";
    }
    function tick() { paint(); raf = window.requestAnimationFrame(tick); }
    function label() {
      if (play) {
        play.textContent = v.paused ? "Play" : "Pause";
        play.setAttribute("aria-label", (v.paused ? "Play " : "Pause ") + (fig.getAttribute("data-name") || "video"));
      }
    }
    v.addEventListener("play", function () { window.cancelAnimationFrame(raf); tick(); label(); });
    v.addEventListener("pause", function () { window.cancelAnimationFrame(raf); paint(); label(); });
    v.addEventListener("seeked", paint);
    v.addEventListener("loadedmetadata", paint);
    function toggle() { if (v.paused) v.play(); else v.pause(); }
    if (play) play.addEventListener("click", toggle);
    v.addEventListener("click", toggle);  // the picture itself plays and pauses
    if (sound) sound.addEventListener("click", function () {
      v.muted = !v.muted;
      sound.textContent = v.muted ? "Sound on" : "Sound off";
      sound.setAttribute("aria-pressed", String(!v.muted));
    });
    if (score) score.addEventListener("click", function (e) {
      var track = score.querySelector(".track");
      if (!track || !dur) return;
      var r = track.getBoundingClientRect();
      var p = (e.clientX - r.left) / r.width;
      if (p < 0 || p > 1) return;
      v.currentTime = Math.min(p * dur, (v.duration || dur) - 0.05);
      paint();
    });
    label();
    paint();
  });

  // Contact sheets and screenshots open in a dialog over the page instead of leaving it. The picture fits the window;
  // clicking it shows it at full size, scrolled. Esc, the close button or a click outside it closes the dialog.
  var sheets = document.querySelectorAll(".sheet a[href]");
  if (sheets.length && window.HTMLDialogElement) {
    var dlg = document.createElement("dialog");
    dlg.className = "viewer";
    dlg.setAttribute("aria-label", "Image");
    dlg.innerHTML = '<div class="viewer-bar"><span class="viewer-cap"></span><span class="viewer-hint">Tap the picture for full size</span><a class="viewer-open" target="_blank" rel="noopener">Open file</a>' +
      '<button type="button" class="viewer-close" aria-label="Close">Close</button></div>' +
      '<div class="viewer-pic"><img alt=""></div>';
    document.body.appendChild(dlg);
    var pic = dlg.querySelector(".viewer-pic"), big = pic.querySelector("img");
    dlg.querySelector(".viewer-close").addEventListener("click", function () { dlg.close(); });
    dlg.addEventListener("click", function (e) { if (e.target === dlg) dlg.close(); });
    big.addEventListener("click", function () {
      var full = pic.classList.toggle("full");
      big.title = full ? "Click to fit the window" : "Click for full size";
    });
    dlg.addEventListener("close", function () { big.removeAttribute("src"); });
    Array.prototype.forEach.call(sheets, function (a) {
      a.addEventListener("click", function (e) {
        if (e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;  // a new tab, as asked
        e.preventDefault();
        var img = a.querySelector("img"), cap = a.closest("figure") && a.closest("figure").querySelector("figcaption");
        pic.classList.remove("full");
        big.src = a.getAttribute("href");
        big.alt = img ? img.alt : "";
        big.title = "Click for full size";
        dlg.querySelector(".viewer-cap").textContent = cap ? cap.textContent.replace(/\s*Open the image for full size\.?\s*$/, "") : "";
        dlg.querySelector(".viewer-open").href = a.getAttribute("href");
        dlg.showModal();
      });
    });
  }

  // A group of videos played together from the start ([data-play-all] before a [data-group]): all restart and play
  // in step, and the button pauses them all. It names what it will do next.
  Array.prototype.forEach.call(document.querySelectorAll("[data-play-all]"), function (btn) {
    var group = btn.parentNode.nextElementSibling;
    if (!group || !group.hasAttribute("data-group")) return;
    var vids = Array.prototype.slice.call(group.querySelectorAll("video"));
    var synced = false;  // started together by this button, and neither paused since
    function playing() { return vids.some(function (v) { return !v.paused; }); }
    function label() { btn.textContent = synced && playing() ? "Pause both" : "Play both from the start"; }
    btn.addEventListener("click", function () {
      if (synced && playing()) {
        synced = false;
        vids.forEach(function (v) { v.pause(); });
      } else {
        // seek both to the start without pausing them: a pause would arrive later and end the sync
        vids.forEach(function (v) { v.currentTime = 0; });
        vids.forEach(function (v) { if (v.paused) v.play(); });
        synced = true;
      }
      label();
    });
    vids.forEach(function (v) {
      v.addEventListener("play", label);
      v.addEventListener("pause", function () { synced = false; label(); });
    });
    label();
  });
})();
