const $ = (q) => document.querySelector(q);
$("#send").addEventListener("click", async () => {
  const f = $("#file").files[0];
  if (!f) { $("#status").textContent = "Pick a file first."; return; }
  $("#status").textContent = "Uploading...";
  $("#out").innerHTML = "";

  const fd = new FormData();
  fd.append("file", f, f.name);

  const headers = {};
  const tok = $("#token").value.trim();
  if (tok) headers["X-Upload-Token"] = tok;

  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd, headers });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      $("#status").textContent = "Server error.";
      $("#out").innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
      return;
    }
    $("#status").textContent = "Done.";
    $("#out").innerHTML = `
      <div>File: <b>${data.file || ""}</b></div>
      <div>Time: <code>${data.timestamp?.local_iso || ""}</code></div>
      <div>Notion: <code>${data.notion_page_id || "—"}</code></div>
      <h3>Parsed</h3>
      <pre>${JSON.stringify(data.parsed, null, 2)}</pre>
    `;
  } catch (e) {
    $("#status").textContent = "Network error.";
    $("#out").innerHTML = `<pre>${String(e)}</pre>`;
  }
});
