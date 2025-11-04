// src/App.js
import React, { useState, useMemo } from "react";
import {
  Search, Download, RefreshCw, Upload, FileText, Mail, BarChart3, Settings,
  Users, CheckCircle, AlertTriangle, XCircle, Clock, TrendingUp, MapPin,
  Phone, Hospital, Award
} from "lucide-react";
import "./App.css"; // keep if you have it (can be empty). Ensure utilities-shim.css is imported LAST in index.js

/* ---------- Minimal API client (same-origin or explicit base) ---------- */
const API_BASE = "http://127.0.0.1:8000"; // change to your backend URL if needed

async function uploadAndValidateCsv(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/validate-csv`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`CSV validate failed: ${res.status}`);
  return res.json(); // { rows: [...] }
}

async function processOcr(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/ocr`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`OCR failed: ${res.status}`);
  return res.json();
}

/* ---------- Mock data (used until CSV is uploaded) ---------- */
const generateMockData = () => {
  const statuses = [
    "VERIFIED_OK (Google + NPI Registry)",
    "VERIFIED_OK (Google only)",
    "NEEDS_UPDATE (Google)",
    "NEEDS_MANUAL_REVIEW",
  ];
  const cities = ["Mumbai","Delhi","Bangalore","Chennai","Kolkata","Hyderabad","Pune","Ahmedabad"];
  const specializations = ["Cardiology","Orthopedics","General Medicine","Pediatrics","ENT","Dermatology","Neurology","Gastroenterology"];

  return Array.from({ length: 50 }, (_, i) => ({
    provider_id: `P${String(i + 1).padStart(4, "0")}`,
    name: `Dr. ${["Rajesh","Priya","Amit","Sneha","Vikram","Anita"][i % 6]} ${["Kumar","Singh","Patel","Sharma","Reddy","Gupta"][i % 6]}`,
    phone: `98765${String(i).padStart(5, "0")}`,
    city: cities[i % cities.length],
    specialization: specializations[i % specializations.length],
    status: statuses[i % statuses.length],
    confidence_score: 30 + Math.floor(Math.random() * 70),
    suggested_phone: Math.random() > 0.5 ? `98765${String(i + 1).padStart(5, "0")}` : null,
    google_phone: Math.random() > 0.3 ? `98765${String(i).padStart(5, "0")}` : null,
    registry_source: ["NPI Registry", "State Medical Council", "Google Maps"][i % 3],
  }));
};

/* ---------- Main App ---------- */
const App = () => {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [searchQuery, setSearchQuery] = useState("");
  const [minConfidence, setMinConfidence] = useState(0);
  const [selectedProvider, setSelectedProvider] = useState(null);
  const [ocrFile, setOcrFile] = useState(null);

  const [providers, setProviders] = useState(generateMockData());
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  // (1) OCR result panel state
  const [ocrResult, setOcrResult] = useState(null);

  // (1) Provider Search tab specific filters
  const [searchCity, setSearchCity] = useState("");
  const [searchStatus, setSearchStatus] = useState("");
  const [searchSpec, setSearchSpec] = useState("");

  // filtering for Validation Results & Dashboard
  const filteredProviders = useMemo(() => {
    return providers.filter(
      (p) =>
        p.confidence_score >= minConfidence &&
        (searchQuery === "" ||
          p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          p.provider_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
          (p.city || "").toLowerCase().includes(searchQuery.toLowerCase()))
    );
  }, [providers, minConfidence, searchQuery]);

  // stats
  const stats = useMemo(() => {
    const total = filteredProviders.length;
    const verified = filteredProviders.filter((p) => p.confidence_score >= 80).length;
    const needsUpdate = filteredProviders.filter((p) => (p.status || "").includes("UPDATE")).length;
    const needsReview = filteredProviders.filter((p) => (p.status || "").includes("REVIEW")).length;
    const avgConfidence =
      filteredProviders.reduce((acc, p) => acc + (p.confidence_score || 0), 0) / (total || 1);
    return { total, verified, needsUpdate, needsReview, avgConfidence };
  }, [filteredProviders]);

  // confidence histogram
  const confidenceDistribution = useMemo(() => {
    const bins = { "0-49": 0, "50-69": 0, "70-79": 0, "80-89": 0, "90-100": 0 };
    filteredProviders.forEach((p) => {
      const s = p.confidence_score || 0;
      if (s < 50) bins["0-49"]++;
      else if (s < 70) bins["50-69"]++;
      else if (s < 80) bins["70-79"]++;
      else if (s < 90) bins["80-89"]++;
      else bins["90-100"]++;
    });
    return bins;
  }, [filteredProviders]);

  // colors
  const getStatusColor = (status) => {
    if (!status) return "chip chip--info";
    if (status.includes("VERIFIED")) return "chip chip--ok";
    if (status.includes("UPDATE")) return "chip chip--warn";
    if (status.includes("REVIEW")) return "chip chip--error";
    return "chip chip--info";
  };
  const getConfidenceColor = (score) => {
    if (score >= 90) return "text-green-600";
    if (score >= 70) return "text-amber-600";
    return "text-red-600";
  };

  /* ---------------- Tabs ---------------- */
  const renderDashboard = () => (
    <div className="space-y-6">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <StatCard icon={Users} label="Total Providers" value={stats.total.toLocaleString()} color="bg-gradient-to-br from-blue-500 to-blue-600" />
        <StatCard icon={CheckCircle} label="High Confidence" value={stats.verified.toLocaleString()} subtitle={`${((stats.verified/(stats.total||1))*100).toFixed(1)}%`} color="bg-gradient-to-br from-green-500 to-green-600" />
        <StatCard icon={AlertTriangle} label="Needs Update" value={stats.needsUpdate.toLocaleString()} subtitle={`${((stats.needsUpdate/(stats.total||1))*100).toFixed(1)}%`} color="bg-gradient-to-br from-amber-500 to-amber-600" />
        <StatCard icon={Clock} label="Manual Review" value={stats.needsReview.toLocaleString()} subtitle={`${((stats.needsReview/(stats.total||1))*100).toFixed(1)}%`} color="bg-gradient-to-br from-red-500 to-red-600" />
        <StatCard icon={TrendingUp} label="Avg Confidence" value={`${stats.avgConfidence.toFixed(1)}%`} color="bg-gradient-to-br from-purple-500 to-purple-600" />
      </div>

      {/* ROI Metrics */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">ROI & Business Impact</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <MetricBox label="Time Saved" value="240 hrs" change="+95% reduction" positive />
          <MetricBox label="Cost Savings" value="$6,000" change="vs manual validation" />
          <MetricBox label="Speed Improvement" value="144x" change="automated processing" positive />
          <MetricBox label="Accuracy Rate" value="95.2%" change="validation accuracy" positive />
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Confidence Distribution</h3>
          <div className="space-y-3">
            {Object.entries(confidenceDistribution).map(([range, count]) => (
              <div key={range} className="flex items-center gap-3">
                <span className="text-sm font-medium text-gray-600 w-20">{range}%</span>
                <div className="flex-1 bg-gray-100 rounded-full h-8 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-blue-600 h-full flex items-center justify-end pr-2 text-white text-xs font-medium transition-all duration-500"
                    style={{ width: `${(stats.total ? (count / stats.total) * 100 : 0)}%` }}
                  >
                    {count > 0 && count}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Top Cities by Provider Count</h3>
          <div className="space-y-3">
            {Object.entries(
              filteredProviders.reduce((acc, p) => {
                acc[p.city] = (acc[p.city] || 0) + 1;
                return acc;
              }, {})
            )
              .sort((a, b) => b[1] - a[1])
              .slice(0, 6)
              .map(([city, count]) => (
                <div key={city} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <MapPin className="w-4 h-4 text-gray-400" />
                    <span className="text-sm font-medium text-gray-700">{city}</span>
                  </div>
                  <span className="text-sm font-semibold text-gray-900">{count} providers</span>
                </div>
              ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderValidationResults = () => (
    <div className="space-y-4">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-center justify-between">
        <span className="text-sm text-blue-800">
          Displaying <strong>{filteredProviders.length.toLocaleString()}</strong> of{" "}
          <strong>{providers.length.toLocaleString()}</strong> providers
        </span>
        <button
          className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors flex items-center gap-2"
          onClick={() => {
            // quick CSV export of current filtered rows
            const cols = ["provider_id","name","phone","city","status","confidence_score"];
            const rows = filteredProviders.map((p) => cols.map((c) => `"${(p[c] ?? "").toString().replace(/"/g, '""')}"`).join(","));
            const csv = [cols.join(","), ...rows].join("\n");
            const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url; a.download = "validation_results.csv"; a.click();
            URL.revokeObjectURL(url);
          }}
        >
          <Download className="w-4 h-4" />
          Export Results
        </button>
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Provider ID</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Name</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Phone</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">City</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Status</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Confidence</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {filteredProviders.slice(0, 50).map((provider) => (
                <tr key={provider.provider_id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-4 py-3 text-sm font-medium text-gray-900">{provider.provider_id}</td>
                  <td className="px-4 py-3 text-sm text-gray-700">{provider.name}</td>
                  <td className="px-4 py-3 text-sm text-gray-700 font-mono">{provider.phone}</td>
                  <td className="px-4 py-3 text-sm text-gray-700">{provider.city}</td>
                  <td className="px-4 py-3">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getStatusColor(provider.status)}`}>
                      {(provider.status || "").split(" ")[0] || "—"}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`text-sm font-semibold ${getConfidenceColor(provider.confidence_score ?? 0)}`}>
                      {(provider.confidence_score ?? 0)}%
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <button
                      onClick={() => setSelectedProvider(provider)}
                      className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                    >
                      View Details
                    </button>
                  </td>
                </tr>
              ))}
              {filteredProviders.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-6 text-center text-sm text-gray-600">
                    No results. Adjust filters or upload a CSV.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {/* (4) Footer showing counts */}
        <div className="px-4 py-2 text-xs text-gray-500 border-t border-gray-200">
          Showing {Math.min(50, filteredProviders.length)} of {filteredProviders.length.toLocaleString()} (total {providers.length.toLocaleString()})
        </div>
      </div>
    </div>
  );

  const renderOCR = () => (
    <div className="space-y-6">
      <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg p-8 text-white">
        <h3 className="text-2xl font-bold mb-2">OCR Document Processing</h3>
        <p className="text-purple-100">
          Upload medical pamphlets, business cards, or provider documents for automated information extraction
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          {/* Make the whole area clickable using <label> */}
          <label className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-purple-500 transition-colors cursor-pointer vs-dropzone">
            <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-sm font-medium text-gray-700 mb-1">Click to upload or drag and drop</p>
            <p className="text-xs text-gray-500">JPG, PNG, PDF, AVIF, WEBP (max ~10MB)</p>
            <input
              type="file"
              accept=".jpg,.jpeg,.png,.pdf,.avif,.webp"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) setOcrFile(f);
              }}
              hidden
            />
          </label>

          {ocrFile && (
            <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <FileText className="w-5 h-5 text-green-600" />
                  <span className="text-sm font-medium text-green-900">{ocrFile.name}</span>
                </div>
                <button
                  className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700"
                  onClick={async () => {
                    // (2) replaced handler: set panel instead of alert
                    if (!ocrFile) return;
                    setLoading(true);
                    setErrorMsg("");
                    try {
                      const result = await processOcr(ocrFile);
                      setOcrResult(result); // <-- show panel
                    } catch (err) {
                      console.error(err);
                      setErrorMsg(err.message || "OCR failed");
                    } finally {
                      setLoading(false);
                    }
                  }}
                >
                  {loading ? "Processing…" : "Process Document"}
                </button>
              </div>
            </div>
          )}

          {/* (2) OCR Result panel */}
          {ocrResult && (
            <div className="mt-4 p-4 bg-white rounded-lg border border-gray-200">
              <h4 className="font-semibold text-gray-800 mb-3">
                OCR Result (Best: {ocrResult.best_preprocessing_method})
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-gray-500 mb-1">Doctor Name</div>
                  <div className="font-medium">{ocrResult?.parsed_information?.doctor_name || "—"}</div>
                </div>
                <div>
                  <div className="text-gray-500 mb-1">Credentials</div>
                  <div className="font-medium">{ocrResult?.parsed_information?.credentials || "—"}</div>
                </div>
                <div>
                  <div className="text-gray-500 mb-1">Hospital</div>
                  <div className="font-medium">{ocrResult?.parsed_information?.hospital_name || "—"}</div>
                </div>
                <div>
                  <div className="text-gray-500 mb-1">Specialization</div>
                  <div className="font-medium">{ocrResult?.parsed_information?.specialization || "—"}</div>
                </div>
                <div>
                  <div className="text-gray-500 mb-1">Phone Numbers</div>
                  <div className="font-medium">
                    {(ocrResult?.parsed_information?.phone_numbers || []).join(", ") || "—"}
                  </div>
                </div>
                <div>
                  <div className="text-gray-500 mb-1">Email</div>
                  <div className="font-medium">{ocrResult?.parsed_information?.email || "—"}</div>
                </div>
                <div className="md:col-span-2">
                  <div className="text-gray-500 mb-1">Conditions Treated</div>
                  <div className="font-medium">
                    {(ocrResult?.parsed_information?.conditions_treated || []).slice(0,6).join(" • ") || "—"}
                  </div>
                </div>
              </div>

              <details className="mt-4">
                <summary className="text-sm text-blue-600 cursor-pointer">Show full extracted text</summary>
                <pre className="mt-2 p-3 bg-gray-50 rounded border border-gray-200 text-xs overflow-x-auto">
{ocrResult?.full_text || ""}
                </pre>
              </details>
            </div>
          )}
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h4 className="font-semibold text-gray-800 mb-4">Processing Steps</h4>
          <div className="space-y-3 text-sm text-gray-600">
            <Step n="1" text="Image preprocessing" />
            <Step n="2" text="Multiple OCR methods" />
            <Step n="3" text="Best result selection" />
            <Step n="4" text="Information extraction" />
            <Step n="5" text="Data validation" />
          </div>
        </div>
      </div>
    </div>
  );

  // (3) Provider Search tab
  const renderProviderSearch = () => {
    // Build source lists from current data
    const cities = Array.from(new Set(providers.map(p => p.city).filter(Boolean))).sort();
    const statuses = Array.from(new Set(providers.map(p => p.status).filter(Boolean))).sort();
    const specs = Array.from(new Set(providers.map(p => p.specialization).filter(Boolean))).sort();

    const results = providers.filter(p => {
      const matchesQuery =
        !searchQuery ||
        p.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.provider_id?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.city?.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesCity = !searchCity || p.city === searchCity;
      const matchesStatus = !searchStatus || p.status === searchStatus;
      const matchesSpec = !searchSpec || p.specialization === searchSpec;
      const matchesConf = (p.confidence_score ?? 0) >= minConfidence;
      return matchesQuery && matchesCity && matchesStatus && matchesSpec && matchesConf;
    });

    return (
      <div className="space-y-4">
        {/* Filters row */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="relative">
              <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search name, ID or city…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-9 pr-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <select
              value={searchCity}
              onChange={(e) => setSearchCity(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
            >
              <option value="">All Cities</option>
              {cities.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
            <select
              value={searchSpec}
              onChange={(e) => setSearchSpec(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
            >
              <option value="">All Specializations</option>
              {specs.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            <select
              value={searchStatus}
              onChange={(e) => setSearchStatus(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
            >
              <option value="">All Statuses</option>
              {statuses.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>

        {/* Results */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Provider ID</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Name</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">City</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Specialization</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Status</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Confidence</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {results.slice(0, 50).map((p) => (
                  <tr key={p.provider_id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">{p.provider_id}</td>
                    <td className="px-4 py-3 text-sm text-gray-700">{p.name}</td>
                    <td className="px-4 py-3 text-sm text-gray-700">{p.city}</td>
                    <td className="px-4 py-3 text-sm text-gray-700">{p.specialization || "—"}</td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getStatusColor(p.status)}`}>
                        {(p.status || "").split(" ")[0] || "—"}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`text-sm font-semibold ${getConfidenceColor(p.confidence_score ?? 0)}`}>
                        {(p.confidence_score ?? 0)}%
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <button className="text-blue-600 hover:text-blue-800 text-sm font-medium" onClick={() => setSelectedProvider(p)}>
                        View
                      </button>
                    </td>
                  </tr>
                ))}
                {results.length === 0 && (
                  <tr><td colSpan={7} className="px-4 py-6 text-center text-sm text-gray-600">No matches.</td></tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="px-4 py-2 text-xs text-gray-500 border-t border-gray-200">
            Showing {Math.min(50, results.length)} of {results.length.toLocaleString()} results
          </div>
        </div>
      </div>
    );
  };

  const renderAnalytics = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">System Performance</h3>
          <div className="grid grid-cols-2 gap-4">
            <Tile label="Providers/Hour" value="500+" scheme="blue" />
            <Tile label="Accuracy Rate" value="95.2%" scheme="green" />
            <Tile label="Cost/Provider" value="$0.05" scheme="purple" />
            <Tile label="Avg Time" value="0.9s" scheme="amber" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Data Source Reliability</h3>
          <div className="space-y-3">
            <ReliabilityBar label="Google Maps" percentage={78} />
            <ReliabilityBar label="Medical Registry" percentage={65} />
            <ReliabilityBar label="NPI Database" percentage={45} />
            <ReliabilityBar label="Combined Sources" percentage={92} />
          </div>
        </div>
      </div>
    </div>
  );

  /* ---------------- Render ---------------- */
  return (
    <div className="min-h-screen bg-gray-50">
      {/* tiny CSS for chips so status looks right without Tailwind */}
      <style>{`
        .chip{display:inline-flex;align-items:center;padding:2px 10px;border-radius:9999px;font-size:12px;font-weight:600;border:1px solid}
        .chip--ok{color:#166534;background:#ecfdf5;border-color:#bbf7d0}
        .chip--warn{color:#92400e;background:#fffbeb;border-color:#fde68a}
        .chip--error{color:#991b1b;background:#fef2f2;border-color:#fecaca}
        .chip--info{color:#1d4ed8;background:#eff6ff;border-color:#bfdbfe}
        label.vs-dropzone{display:block;width:100%;}
      `}</style>

      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">VaidyaSetu</h1>
              <p className="text-xs text-gray-500">Automated Healthcare Directory Management</p>
            </div>
            <div className="flex items-center gap-3">
              <button className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors">
                <Settings className="w-5 h-5" />
              </button>
              <button
                className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                onClick={() => {
                  // simple refresh: reload sample
                  setProviders(generateMockData());
                }}
              >
                <RefreshCw className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex gap-6">
          {/* Sidebar */}
          <aside className="w-64 flex-shrink-0">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 sticky top-24">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Control Center</h3>

              <div className="space-y-2 mb-6">
                <div className="p-3 bg-gray-50 rounded-lg">
                  <div className="text-xs text-gray-500">Total Providers</div>
                  <div className="text-lg font-bold text-gray-900">{providers.length}</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg">
                  <div className="text-xs text-gray-500">Avg Confidence</div>
                  <div className="text-lg font-bold text-gray-900">{stats.avgConfidence.toFixed(1)}%</div>
                </div>
              </div>

              <div className="space-y-3 mb-6">
                <div>
                  <label className="text-xs font-medium text-gray-700 mb-2 block">Search</label>
                  <div className="relative">
                    <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
                    <input
                      type="text"
                      placeholder="Search..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="w-full pl-9 pr-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-xs font-medium text-gray-700 mb-2 block">
                    Min Confidence: {minConfidence}%
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    step="5"
                    value={minConfidence}
                    onChange={(e) => setMinConfidence(Number(e.target.value))}
                    className="w-full"
                  />
                </div>

                {/* CSV uploader */}
                <div>
                  <label className="text-xs font-medium text-gray-700 mb-2 block">Upload CSV</label>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={async (e) => {
                      const f = e.target.files?.[0];
                      if (!f) return;
                      setErrorMsg("");
                      setLoading(true);
                      try {
                        const data = await uploadAndValidateCsv(f);
                        setProviders(data.rows || []);
                      } catch (err) {
                        console.error(err);
                        setErrorMsg(err.message || "Upload failed");
                      } finally {
                        setLoading(false);
                      }
                    }}
                  />
                  {loading && <div className="text-xs text-gray-500 mt-1">Processing…</div>}
                  {errorMsg && <div className="text-xs text-red-600 mt-1">{errorMsg}</div>}
                </div>
              </div>

              <div className="pt-4 border-t border-gray-200">
                <button
                  className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors mb-2"
                  onClick={() => setProviders(generateMockData())}
                >
                  Load Sample Data
                </button>
                <button
                  className="w-full px-4 py-2 bg-gray-100 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-200 transition-colors"
                  onClick={() => {
                    // simple client-side export of ALL providers
                    const cols = ["provider_id","name","phone","city","status","confidence_score"];
                    const rows = providers.map((p) => cols.map((c) => `"${(p[c] ?? "").toString().replace(/"/g, '""')}"`).join(","));
                    const csv = [cols.join(","), ...rows].join("\n");
                    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url; a.download = "report.csv"; a.click();
                    URL.revokeObjectURL(url);
                  }}
                >
                  Export Report
                </button>
              </div>
            </div>
          </aside>

          {/* Main Content */}
          <main className="flex-1">
            {/* Tabs */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 mb-6">
              <div className="flex border-b border-gray-200 overflow-x-auto">
                <TabButton icon={BarChart3} label="Dashboard" active={activeTab === "dashboard"} onClick={() => setActiveTab("dashboard")} />
                <TabButton icon={Users} label="Validation Results" active={activeTab === "validation"} onClick={() => setActiveTab("validation")} />
                <TabButton icon={Search} label="Provider Search" active={activeTab === "search"} onClick={() => setActiveTab("search")} />
                <TabButton icon={FileText} label="OCR Processing" active={activeTab === "ocr"} onClick={() => setActiveTab("ocr")} />
                <TabButton icon={Mail} label="Communications" active={activeTab === "communications"} onClick={() => setActiveTab("communications")} />
                <TabButton icon={TrendingUp} label="Analytics" active={activeTab === "analytics"} onClick={() => setActiveTab("analytics")} />
              </div>
            </div>

            {/* Tab Content */}
            <div>
              {activeTab === "dashboard" && renderDashboard()}
              {activeTab === "validation" && renderValidationResults()}
              {/* (3) switch to provider search renderer */}
              {activeTab === "search" && renderProviderSearch()}
              {activeTab === "ocr" && renderOCR()}
              {activeTab === "communications" && (
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
                  <Mail className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Communication Center</h3>
                  <p className="text-gray-600 mb-6">Email generation and provider communications will appear here</p>
                  <button className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700">
                    Generate Emails
                  </button>
                </div>
              )}
              {activeTab === "analytics" && renderAnalytics()}
            </div>
          </main>
        </div>
      </div>

      {/* Provider Detail Modal */}
      {selectedProvider && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
          onClick={() => setSelectedProvider(null)}
        >
          <div
            className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
              <h3 className="text-xl font-semibold text-gray-900">Provider Details</h3>
              <button onClick={() => setSelectedProvider(null)} className="text-gray-400 hover:text-gray-600">
                <XCircle className="w-6 h-6" />
              </button>
            </div>
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <h4 className="text-sm font-semibold text-gray-700 mb-3">Basic Information</h4>
                  <div className="space-y-2 text-sm">
                    <div><span className="text-gray-500">Provider ID:</span> <span className="font-medium">{selectedProvider.provider_id}</span></div>
                    <div><span className="text-gray-500">Name:</span> <span className="font-medium">{selectedProvider.name}</span></div>
                    <div className="flex items-center gap-2"><Phone className="w-4 h-4 text-gray-400" /><span className="font-mono">{selectedProvider.phone}</span></div>
                    <div className="flex items-center gap-2"><MapPin className="w-4 h-4 text-gray-400" /><span>{selectedProvider.city}</span></div>
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-gray-700 mb-3">Practice Details</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2"><Hospital className="w-4 h-4 text-gray-400" /><span>{selectedProvider.specialization}</span></div>
                    <div className="flex items-center gap-2"><Award className="w-4 h-4 text-gray-400" /><span>{selectedProvider.registry_source}</span></div>
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-gray-700 mb-3">Validation Status</h4>
                  <div className="space-y-3">
                    <div className={`px-3 py-2 rounded-lg border text-sm font-medium ${getStatusColor(selectedProvider.status)}`}>
                      {selectedProvider.status}
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <div className="text-xs text-gray-500 mb-1">Confidence Score</div>
                      <div className={`text-2xl font-bold ${getConfidenceColor(selectedProvider.confidence_score)}`}>
                        {selectedProvider.confidence_score}%
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="mt-6 pt-6 border-t border-gray-200 flex gap-3">
                <button className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors">
                  Send Email
                </button>
                <button className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 transition-colors">
                  Mark Verified
                </button>
                <button className="flex-1 px-4 py-2 bg-amber-600 text-white rounded-lg text-sm font-medium hover:bg-amber-700 transition-colors">
                  Flag Issue
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

/* ---------- Reusable components ---------- */
const StatCard = ({ icon: Icon, label, value, subtitle, color }) => (
  <div className={`${color} rounded-lg p-6 text-white shadow-lg`}>
    <Icon className="w-8 h-8 mb-3 opacity-90" />
    <div className="text-3xl font-bold mb-1">{value}</div>
    <div className="text-sm opacity-90">{label}</div>
    {subtitle && <div className="text-xs mt-1 opacity-75">{subtitle}</div>}
  </div>
);

const MetricBox = ({ label, value, change, positive }) => (
  <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
    <div className="text-2xl font-bold text-gray-900 mb-1">{value}</div>
    <div className="text-xs text-gray-600 mb-2">{label}</div>
    {change && (
      <div className={`text-xs font-medium ${positive ? "text-green-600" : "text-gray-600"}`}>{change}</div>
    )}
  </div>
);

const TabButton = ({ icon: Icon, label, active, onClick }) => (
  <button
    onClick={onClick}
    className={`flex items-center gap-2 px-6 py-3 text-sm font-medium transition-colors border-b-2 whitespace-nowrap ${
      active ? "border-blue-600 text-blue-600 bg-blue-50" : "border-transparent text-gray-600 hover:text-gray-900 hover:bg-gray-50"
    }`}
  >
    <Icon className="w-4 h-4" />
    {label}
  </button>
);

const ReliabilityBar = ({ label, percentage }) => (
  <div>
    <div className="flex items-center justify-between mb-1">
      <span className="text-sm font-medium text-gray-700">{label}</span>
      <span className="text-sm font-semibold text-gray-900">{percentage}%</span>
    </div>
    <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
      <div
        className="bg-gradient-to-r from-blue-500 to-blue-600 h-full transition-all duration-500"
        style={{ width: `${percentage}%` }}
      />
    </div>
  </div>
);

const Step = ({ n, text }) => (
  <div className="flex items-start gap-2">
    <div className="w-5 h-5 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center text-xs font-semibold mt-0.5">
      {n}
    </div>
    <span>{text}</span>
  </div>
);

const Tile = ({ label, value, scheme }) => {
  const map = {
    blue: ["from-blue-50", "to-blue-100", "text-blue-700", "text-blue-600"],
    green: ["from-green-50", "to-green-100", "text-green-700", "text-green-600"],
    purple: ["from-purple-50", "to-purple-100", "text-purple-700", "text-purple-600"],
    amber: ["from-amber-50", "to-amber-100", "text-amber-700", "text-amber-600"],
  }[scheme || "blue"];

  return (
    <div className={`p-4 bg-gradient-to-br ${map[0]} ${map[1]} rounded-lg`}>
      <div className={`text-2xl font-bold ${map[2]}`}>{value}</div>
      <div className={`text-xs ${map[3]} mt-1`}>{label}</div>
    </div>
  );
};

export default App;
