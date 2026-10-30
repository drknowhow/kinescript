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
    if (play) play.addEventListener("click", function () { if (v.paused) v.play(); else v.pause(); });
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
})();
