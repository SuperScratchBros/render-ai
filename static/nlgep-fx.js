/* NLGEP FX — Opera GX style sound effects for every button, card and keystroke.
   The sounds are the same tiny WebAudio synth used by the GX skin (no audio files needed).
   Loaded on the home page, the AI page and the Chat page. */
(function () {
  "use strict";
  if (window.__nlgepFx) return;
  window.__nlgepFx = true;

  var store = null;
  try { store = window.localStorage; } catch (e) { store = null; }

  var state = {
    on: !(store && store.getItem("nlgep-sfx") === "off"),
    vol: 0.5,
    ctx: null,
    pack: null,
    unlocked: false,
    lastHoverAt: 0
  };

  /* ---------- the GX synth ---------- */
  function synthBuf(c, dur, fn) {
    var sr = c.sampleRate, n = Math.max(1, Math.floor(sr * dur));
    var buf = c.createBuffer(1, n, sr), d = buf.getChannelData(0);
    for (var i = 0; i < n; i++) {
      var t = i / sr, env = Math.pow(1 - i / n, 2.2);
      d[i] = fn(t, i / n) * env;
    }
    return buf;
  }
  function tone(f0, f1, type) {
    return function (t, p) {
      var f = f0 + (f1 - f0) * p, ph = 2 * Math.PI * f * t;
      if (type === "square") return Math.sign(Math.sin(ph)) * 0.5;
      if (type === "tri") return Math.asin(Math.sin(ph)) * (2 / Math.PI) * 0.8;
      return Math.sin(ph);
    };
  }
  function build(c) {
    var P = {};
    P.CLICK = [synthBuf(c, 0.05, tone(1245, 1865, "tri"))];
    P.IMPORTANT_CLICK = [synthBuf(c, 0.07, tone(880, 1320, "square"))];
    P.HOVER = [synthBuf(c, 0.025, tone(2093, 2093, "sine"))];
    P.HOVER_UP = [synthBuf(c, 0.025, tone(2400, 2400, "sine"))];
    P.TAB_INSERT = [synthBuf(c, 0.09, tone(587, 1175, "tri"))];
    P.TAB_CLOSE = [synthBuf(c, 0.09, tone(880, 440, "tri"))];
    P.SWITCH_TOGGLE = [synthBuf(c, 0.06, tone(700, 980, "tri"))];
    P.FEATURE_SWITCH_ON = [synthBuf(c, 0.08, tone(520, 1040, "tri"))];
    P.FEATURE_SWITCH_OFF = [synthBuf(c, 0.08, tone(1040, 520, "tri"))];
    P.LEVEL_UPGRADE = [synthBuf(c, 0.34, function (t) {
      var seq = [523.25, 659.25, 783.99, 1046.5];
      var f = seq[Math.min(3, Math.floor(t / 0.085))];
      return Math.asin(Math.sin(2 * Math.PI * f * t)) * (2 / Math.PI) * 0.7;
    })];
    P.TYPING_LETTER = [
      synthBuf(c, 0.018, tone(2200, 1800, "sine")),
      synthBuf(c, 0.018, tone(2500, 2000, "sine")),
      synthBuf(c, 0.018, tone(1900, 1600, "sine"))
    ];
    P.TYPING_ENTER = [synthBuf(c, 0.05, tone(1200, 600, "tri"))];
    P.TYPING_SPACE = [synthBuf(c, 0.022, tone(1600, 1400, "sine"))];
    P.TYPING_BACKSPACE = [synthBuf(c, 0.03, tone(640, 500, "sine"))];
    return P;
  }
  function audioCtx() {
    if (!state.ctx) {
      var C = window.AudioContext || window.webkitAudioContext;
      if (!C) return null;
      try { state.ctx = new C(); } catch (e) { return null; }
    }
    if (state.ctx.state === "suspended") { try { state.ctx.resume(); } catch (e) {} }
    return state.ctx;
  }
  function play(ev) {
    if (!state.on || !state.unlocked) return;
    try {
      var c = audioCtx();
      if (!c || c.state === "suspended") return;
      if (!state.pack) state.pack = build(c);
      var arr = state.pack[ev];
      if (!arr || !arr.length) return;
      var s = c.createBufferSource(), g = c.createGain();
      s.buffer = arr[Math.floor(Math.random() * arr.length)];
      g.gain.value = state.vol;
      s.connect(g); g.connect(c.destination); s.start();
    } catch (e) {}
  }
  window.NLGEP_FX = { play: play };

  /* ---------- what counts as a button / a card ---------- */
  var BUTTONS = 'button,[role="button"],input[type="button"],input[type="submit"],input[type="reset"],select,summary,' +
    'a.tool-btn,a.nl-nav,a.nb,.tool-btn,.nl-nav,.nb,.navbtn,.tool,.send';
  var CARDS = '.hub-card,.nl-flip,#nlHomeClockCard,#nlHomeTitleCard,.family-card,.model-detail-card,' +
    '.usage-card,.usage-item,.chat-entry,.msg,.message,.user,.nl-pill';
  var ALL = BUTTONS + "," + CARDS;

  function target(e) {
    var t = e.target;
    if (!t || !t.closest) return null;
    var el = t.closest(ALL);
    if (el && (el.disabled || el.getAttribute("aria-disabled") === "true")) return null;
    return el;
  }
  function isCard(el) { return el.matches(CARDS) && !el.matches(BUTTONS); }

  /* ---------- unlock audio on the first user gesture (browser autoplay rules) ---------- */
  function unlock() {
    if (state.unlocked) return;
    state.unlocked = true;
    audioCtx();
  }
  ["pointerdown", "keydown", "touchstart", "mousedown"].forEach(function (n) {
    document.addEventListener(n, unlock, { capture: true, passive: true });
  });

  /* ---------- hover ---------- */
  var lastEl = null;
  document.addEventListener("mouseover", function (e) {
    var el = target(e);
    if (el && el !== lastEl) {
      lastEl = el;
      var now = Date.now();
      if (now - state.lastHoverAt > 35) { state.lastHoverAt = now; play(isCard(el) ? "HOVER_UP" : "HOVER"); }
    } else if (!el) {
      lastEl = null;
    }
  }, true);

  /* ---------- click ---------- */
  document.addEventListener("click", function (e) {
    var tog = e.target && e.target.closest ? e.target.closest("[data-sfx-toggle]") : null;
    if (tog) {
      state.on = !state.on;
      if (store) { try { store.setItem("nlgep-sfx", state.on ? "on" : "off"); } catch (x) {} }
      paint();
      play(state.on ? "FEATURE_SWITCH_ON" : "FEATURE_SWITCH_OFF");
      return;
    }
    var el = target(e);
    if (!el) return;
    play(el.matches(".send,.primary,.tool-btn.primary,input[type=\"submit\"]") ? "IMPORTANT_CLICK" : "CLICK");
  }, true);

  /* ---------- typing ---------- */
  var TEXT_TYPES = { "": 1, text: 1, password: 1, search: 1, email: 1, url: 1, tel: 1, number: 1 };
  function isTextTarget(t) {
    if (!t) return false;
    if (t.tagName === "TEXTAREA") return true;
    if (t.tagName === "INPUT") return !!TEXT_TYPES[(t.type || "").toLowerCase()];
    return !!t.isContentEditable;
  }
  document.addEventListener("keydown", function (e) {
    if (!isTextTarget(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
    var k = e.key, ev;
    if (k === "Enter") ev = "TYPING_ENTER";
    else if (k === " " || k === "Spacebar") ev = "TYPING_SPACE";
    else if (k === "Backspace" || k === "Delete") ev = "TYPING_BACKSPACE";
    else if (k && k.length === 1) ev = "TYPING_LETTER";
    else return;
    play(ev);
  }, true);

  /* ---------- sound on/off buttons ---------- */
  function paint() {
    var list = document.querySelectorAll("[data-sfx-toggle]");
    for (var i = 0; i < list.length; i++) {
      var b = list[i], l = b.querySelector(".sfx-label"), ic = b.querySelector("i.sfx-ico");
      b.setAttribute("aria-pressed", state.on ? "true" : "false");
      if (l) l.textContent = state.on ? "Sound: On" : "Sound: Off";
      if (ic) ic.className = "sfx-ico fas " + (state.on ? "fa-volume-high" : "fa-volume-xmark");
      if (!l && !ic) b.textContent = state.on ? "\uD83D\uDD0A" : "\uD83D\uDD07";
    }
  }

  /* ---------- the AI page has no sidebar, so add Home / Chat / Sound to its header ---------- */
  function injectAiNav() {
    var chrome = document.querySelector("header.chrome");
    if (!chrome || chrome.querySelector(".nlgep-nav")) return;
    var wrap = document.createElement("div");
    wrap.className = "nlgep-nav";
    wrap.style.cssText = "display:flex;gap:6px;align-items:center";
    wrap.innerHTML =
      '<a class="nb" href="/" title="NLGEP home" style="text-decoration:none;color:inherit">\u2302 Home</a>' +
      '<a class="nb" href="/chat" title="NLGEP Chat" style="text-decoration:none;color:inherit">Chat</a>' +
      '<button type="button" class="nb" data-sfx-toggle title="Turn sounds on or off"></button>';
    var brand = chrome.querySelector(".brand-chip");
    chrome.insertBefore(wrap, brand ? brand.nextSibling : null);
  }

  function ready() { injectAiNav(); paint(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", ready);
  else ready();
})();
