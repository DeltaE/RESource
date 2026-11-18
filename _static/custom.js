/* =============================================================================
   RESource documentation — subtle motion
   Author: Md Eliasinul Islam
   - Scroll-reveal for content blocks (IntersectionObserver)
   - Slim scroll-progress indicator
   Fully guarded for prefers-reduced-motion and re-run on page navigation.
   Purely additive: loaded via html_js_files, no build/workflow change.
   ========================================================================== */
(function () {
  "use strict";

  var reduce = window.matchMedia &&
               window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  /* ---- Scroll-reveal ------------------------------------------------------ */
  function setupReveal() {
    var article = document.querySelector(".bd-article");
    if (!article) return;

    // Target meaningful, top-level blocks only — keeps motion tasteful.
    var selectors = [
      ".bd-article > section > p",
      ".bd-article > section > .admonition",
      ".bd-article > section > figure",
      ".bd-article > section > .highlight",
      ".bd-article > section > table",
      ".bd-article > section > section > .admonition",
      ".bd-article > section > section > figure"
    ];
    var nodes = article.querySelectorAll(selectors.join(","));

    if (reduce || !("IntersectionObserver" in window)) {
      nodes.forEach(function (n) { n.classList.add("res-visible"); });
      return;
    }

    nodes.forEach(function (n) { n.classList.add("res-reveal"); });

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          e.target.classList.add("res-visible");
          io.unobserve(e.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });

    nodes.forEach(function (n) { io.observe(n); });
  }

  /* ---- Scroll progress bar ------------------------------------------------ */
  function setupProgress() {
    if (reduce) return;
    var bar = document.getElementById("res-progress");
    if (!bar) {
      bar = document.createElement("div");
      bar.id = "res-progress";
      document.body.appendChild(bar);
    }
    var ticking = false;
    function update() {
      var h = document.documentElement;
      var scrolled = h.scrollTop;
      var height = h.scrollHeight - h.clientHeight;
      var pct = height > 0 ? (scrolled / height) * 100 : 0;
      bar.style.width = pct + "%";
      ticking = false;
    }
    window.addEventListener("scroll", function () {
      if (!ticking) { window.requestAnimationFrame(update); ticking = true; }
    }, { passive: true });
    update();
  }

  ready(function () {
    setupReveal();
    setupProgress();
  });
})();
