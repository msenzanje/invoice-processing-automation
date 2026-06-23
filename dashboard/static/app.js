/*
 * Invoice Processing Automation Dashboard — client logic.
 *
 * Renders the audit-trail payload the server embeds at first paint, then polls
 * /api/summary for new records and animates them in. State lives in one `state`
 * object; every interaction (filter, search, expand, theme) re-derives the view from
 * it. No framework — the prototype was React, but the real page is small enough that
 * a template-clone render keeps it dependency-free and fast.
 */
(() => {
  "use strict";

  // ── State ───────────────────────────────────────────────────────────────
  const bootstrap = JSON.parse(document.getElementById("bootstrap-data").textContent);
  const state = {
    data: bootstrap,
    filter: "all",
    query: "",
    expandedId: null,
    newKeys: new Set(), // ids that arrived on the last poll — animated in the feed
  };

  const DECISIONS = ["all", "approved", "needs_review", "rejected"];
  const CHIP_LABELS = { all: "All", approved: "Approved", needs_review: "Needs review", rejected: "Rejected" };

  const $ = (sel) => document.querySelector(sel);
  const rowTemplate = $("#row-template");
  const feedTemplate = $("#feed-template");

  // ── Theme (persisted) ───────────────────────────────────────────────────
  const THEME_KEY = "foundry-ap-theme";
  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    $("#theme-icon").textContent = theme === "dark" ? "☀" : "☾";
    $("#theme-label").textContent = theme === "dark" ? "Light" : "Dark";
  }
  function initTheme() {
    let theme = "light";
    try {
      theme = localStorage.getItem(THEME_KEY) || "light";
    } catch (_) { /* storage blocked — default light */ }
    applyTheme(theme === "dark" ? "dark" : "light");
    $("#theme-toggle").addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      applyTheme(next);
      try { localStorage.setItem(THEME_KEY, next); } catch (_) { /* ignore */ }
    });
  }

  // ── KPI band ────────────────────────────────────────────────────────────
  // Count-up animation mirrors the prototype's eased reveal of the headline numbers.
  function animateValue(el, to, format, duration = 850) {
    const start = performance.now();
    const from = 0;
    function frame(now) {
      const r = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - r, 3);
      el.textContent = format(from + (to - from) * eased);
      if (r < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function renderKpis(animate) {
    const k = state.data.kpis;
    const processedEl = $("#kpi-processed");
    const rateEl = $("#kpi-rate");
    if (animate) {
      animateValue(processedEl, k.processed, (v) => Math.round(v).toLocaleString("en-US"));
      animateValue(rateEl, k.auto_approval_rate * 100, (v) => v.toFixed(1) + "%");
    } else {
      processedEl.textContent = k.processed_fmt;
      rateEl.textContent = k.rate_fmt;
    }
    $("#kpi-processed-delta").textContent = k.processed_fmt;
    $("#kpi-avg").textContent = k.avg_fmt;
    $("#kpi-value").textContent = k.value_fmt;
    // Bar fill animates via CSS transition once width is set.
    requestAnimationFrame(() => { $("#kpi-rate-bar").style.width = k.rate_bar; });
  }

  // ── Filter chips ────────────────────────────────────────────────────────
  function renderChips() {
    const wrap = $("#chips");
    wrap.innerHTML = "";
    for (const id of DECISIONS) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip" + (state.filter === id ? " active" : "");
      btn.innerHTML = `${CHIP_LABELS[id]} <span class="chip-count">${state.data.counts[id] ?? 0}</span>`;
      btn.addEventListener("click", () => {
        state.filter = id;
        renderChips();
        renderRows();
      });
      wrap.appendChild(btn);
    }
  }

  // ── Table rows ──────────────────────────────────────────────────────────
  function filteredRows() {
    const q = state.query.trim().toLowerCase();
    return state.data.rows.filter((row) => {
      if (state.filter !== "all" && row.decision !== state.filter) return false;
      if (q && !(row.vendor.toLowerCase().includes(q) || row.invoice_id.toLowerCase().includes(q))) return false;
      return true;
    });
  }

  function renderRows() {
    const container = $("#rows");
    const rows = filteredRows();
    container.innerHTML = "";
    $("#shown-count").textContent = rows.length;
    $("#rows-empty").hidden = rows.length !== 0;

    for (const row of rows) {
      const node = rowTemplate.content.cloneNode(true);
      const wrap = node.querySelector(".row-wrap");
      wrap.classList.add("d-" + row.decision);
      if (state.expandedId === row.invoice_id) wrap.classList.add("expanded");

      node.querySelector("[data-id]").textContent = row.id_short;
      node.querySelector("[data-time]").textContent = row.time_ago;
      node.querySelector("[data-vendor]").textContent = row.vendor;
      node.querySelector("[data-amount]").textContent = row.amount_fmt;
      node.querySelector("[data-badge-text]").textContent = row.badge_text;
      node.querySelector("[data-payment]").textContent = row.payment_text;

      // Detail panel
      node.querySelector("[data-chain-text]").textContent = row.badge_text;
      node.querySelector("[data-reasoning]").textContent = row.reasoning;
      node.querySelector("[data-critique]").textContent = row.critique;
      node.querySelector("[data-meta-id]").textContent = row.id_short;
      node.querySelector("[data-meta-amount]").textContent = row.amount_fmt;
      node.querySelector("[data-meta-payment]").textContent = row.payment_text;
      node.querySelector("[data-meta-proc]").textContent = row.proc_ms;

      const flagList = node.querySelector("[data-flags]");
      if (!row.flags.length) {
        const none = document.createElement("span");
        none.className = "flag-none";
        none.textContent = "None";
        flagList.appendChild(none);
      } else {
        for (const flag of row.flags) {
          const chip = document.createElement("span");
          chip.className = "flag";
          chip.textContent = flag;
          flagList.appendChild(chip);
        }
      }

      wrap.querySelector("[data-toggle]").addEventListener("click", () => {
        state.expandedId = state.expandedId === row.invoice_id ? null : row.invoice_id;
        renderRows();
      });

      container.appendChild(node);
    }
  }

  // ── Live feed ───────────────────────────────────────────────────────────
  function renderFeed() {
    const container = $("#feed");
    container.innerHTML = "";
    if (!state.data.feed.length) {
      const empty = document.createElement("div");
      empty.className = "feed-empty";
      empty.textContent = "No recent activity. Run the pipeline to see invoices appear here.";
      container.appendChild(empty);
      return;
    }
    for (const item of state.data.feed) {
      const node = feedTemplate.content.cloneNode(true);
      node.querySelector("[data-vendor]").textContent = item.vendor;
      node.querySelector("[data-id]").textContent = item.id_short;
      node.querySelector("[data-amount]").textContent = item.amount_fmt;
      node.querySelector("[data-label]").textContent = item.label;
      // The decision class drives icon/dot/label colours (CSS: .fd-x .feed-card …).
      const wrapper = document.createElement("div");
      wrapper.className = "fd-" + item.decision;
      wrapper.appendChild(node);
      container.appendChild(wrapper);
    }
  }

  // ── Polling ─────────────────────────────────────────────────────────────
  // The audit log is append-only; poll the summary endpoint and re-render only when a
  // newer record exists (cheap timestamp diff), so a quiet pipeline costs nothing.
  const POLL_MS = 4000;
  async function poll() {
    try {
      const res = await fetch("/api/summary", { headers: { "Accept": "application/json" } });
      if (!res.ok) return;
      const next = await res.json();
      if (next.latest_timestamp !== state.data.latest_timestamp || next.total !== state.data.total) {
        const prevIds = new Set(state.data.rows.map((r) => r.invoice_id + r.time_ago));
        state.data = next;
        // Mark genuinely new feed items so the newest one renders as "processing".
        state.newKeys = new Set(
          next.feed.filter((f) => !prevIds.has(f.invoice_id + "")).map((f) => f.invoice_id)
        );
        renderKpis(false);
        renderChips();
        renderRows();
        renderFeed();
      }
    } catch (_) { /* transient network error — try again next tick */ }
  }

  // ── Init ────────────────────────────────────────────────────────────────
  function init() {
    initTheme();
    $("#search").addEventListener("input", (e) => {
      state.query = e.target.value;
      renderRows();
    });
    renderKpis(true);
    renderChips();
    renderRows();
    renderFeed();
    setInterval(poll, POLL_MS);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
