import React, { useState } from 'react';
import { Upload, Activity, ShieldAlert, CheckCircle, AlertTriangle, Download, LayoutDashboard, BarChart3, Image as ImageIcon, Columns } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('inspector');
  const [image, setImage] = useState(null);
  const [file, setFile] = useState(null);

  // Single Inspector State
  const [resultData, setResultData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState("YOLO11s (Attention Enhanced)");
  const [pThresh, setPThresh] = useState(0.5);
  const [gThresh, setGThresh] = useState(0.35);

  // Dual Comparison State
  const [modelLeft, setModelLeft] = useState("YOLO11s (Attention Enhanced)");
  const [modelRight, setModelRight] = useState("RetinaNet (Baseline)");
  const [dualLoading, setDualLoading] = useState(false);
  const [dualResultLeft, setDualResultLeft] = useState(null);
  const [dualResultRight, setDualResultRight] = useState(null);

  const handleImageUpload = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setImage(URL.createObjectURL(selected));
      setResultData(null);
      setDualResultLeft(null);
      setDualResultRight(null);
    }
  };

  // Run Single Inference
  const runInspection = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_name", model);
    formData.append("person_thresh", pThresh);
    formData.append("gear_thresh", gThresh);

    try {
      const response = await fetch("https://trace-nickname-book.ngrok-free.dev/api/inspect", {
        method: "POST",
        headers: {
          "ngrok-skip-browser-warning": "69420", // Bypasses the Ngrok warning screen
        },
        body: formData,
      });
      const data = await response.json();
      setResultData(data);
    } catch (error) {
      console.error("Error connecting to backend:", error);
    }
    setLoading(false);
  };

  // Run Simultaneous Dual Inference
  const runDualInspection = async () => {
    if (!file) return;
    setDualLoading(true);

    const formData1 = new FormData();
    formData1.append("file", file); formData1.append("model_name", modelLeft);
    formData1.append("person_thresh", pThresh); formData1.append("gear_thresh", gThresh);

    const formData2 = new FormData();
    formData2.append("file", file); formData2.append("model_name", modelRight);
    formData2.append("person_thresh", pThresh); formData2.append("gear_thresh", gThresh);

    try {
      const [res1, res2] = await Promise.all([
        fetch("https://trace-nickname-book.ngrok-free.dev/api/inspect", { 
          method: "POST", 
          headers: { "ngrok-skip-browser-warning": "69420" },
          body: formData1 
        }),
        fetch("https://trace-nickname-book.ngrok-free.dev/api/inspect", { 
          method: "POST", 
          headers: { "ngrok-skip-browser-warning": "69420" },
          body: formData2 
        })
      ]);
      setDualResultLeft(await res1.json());
      setDualResultRight(await res2.json());
    } catch (error) {
      console.error("Error running dual inspection:", error);
    }
    setDualLoading(false);
  };

  // Hardcoded Model Specs
  const benchmarkData = [
    { name: "YOLO11s (Attention Enhanced)", params: 9.4, gflops: 21.4, map50: 92.0, fps: 62 },
    { name: "YOLOv8n (Lightweight Baseline)", params: 3.2, gflops: 8.7, map50: 90.0, fps: 132 },
    { name: "RetinaNet (Baseline)", params: 34.0, gflops: 90.0, map50: 81.0, fps: 12 }
  ];

  // Map models to their respective public folders
  const getEvalFolder = (modelName) => {
    if (modelName.includes("11s")) return "/evaluation_output/test_metrics/";
    if (modelName.includes("v8n")) return "/evaluation8n_output/test_metrics/";
    return "/evaluation_retinanet_output/test_metrics/";
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-blue-500/30">
      <div className="fixed inset-0 z-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-blue-900/20 via-slate-950 to-slate-950"></div>

      <div className="relative z-10 max-w-7xl mx-auto p-6">
        <header className="mb-6 p-6 rounded-2xl bg-slate-900/50 backdrop-blur-xl border border-slate-800 shadow-2xl flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-4xl font-extrabold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
              🦺 SafeSight AI
            </h1>
            <p className="text-slate-400 mt-2 text-sm md:text-lg">Real-Time PPE Compliance Monitor</p>
          </div>

          <div className="flex flex-wrap bg-slate-800/50 p-1 rounded-xl border border-slate-700">
            <button onClick={() => setActiveTab('inspector')} className={`flex items-center gap-2 px-3 py-2 rounded-lg font-medium transition-all ${activeTab === 'inspector' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}>
              <LayoutDashboard size={18} /> Inspector
            </button>
            <button onClick={() => setActiveTab('dual')} className={`flex items-center gap-2 px-3 py-2 rounded-lg font-medium transition-all ${activeTab === 'dual' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}>
              <Columns size={18} /> Dual Compare
            </button>
            <button onClick={() => setActiveTab('benchmark')} className={`flex items-center gap-2 px-3 py-2 rounded-lg font-medium transition-all ${activeTab === 'benchmark' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}>
              <BarChart3 size={18} /> Benchmarks
            </button>
            <button onClick={() => setActiveTab('evaluation')} className={`flex items-center gap-2 px-3 py-2 rounded-lg font-medium transition-all ${activeTab === 'evaluation' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}>
              <ImageIcon size={18} /> Proofs
            </button>
          </div>
        </header>

        {/* Global Sidebar for Uploads (Shared between Inspector and Dual) */}
        {(activeTab === 'inspector' || activeTab === 'dual') && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500 mb-6">
            <div className="lg:col-span-1 space-y-6">
              <div className="p-5 rounded-2xl bg-slate-900/50 backdrop-blur-md border border-slate-800">
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-blue-400" /> Settings
                </h2>

                {activeTab === 'inspector' ? (
                  <>
                    <label className="block text-sm text-slate-400 mb-2">Detection Engine</label>
                    <select value={model} onChange={(e) => setModel(e.target.value)} className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 mb-6 text-slate-200 outline-none focus:border-blue-500 transition-colors">
                      <option>YOLO11s (Attention Enhanced)</option>
                      <option>YOLOv8n (Lightweight)</option>
                      <option>RetinaNet (Baseline)</option>
                    </select>
                  </>
                ) : (
                  <div className="space-y-4 mb-6">
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Left Engine</label>
                      <select value={modelLeft} onChange={(e) => setModelLeft(e.target.value)} className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-sm text-slate-200 outline-none">
                        <option>YOLO11s (Attention Enhanced)</option>
                        <option>YOLOv8n (Lightweight)</option>
                        <option>RetinaNet (Baseline)</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Right Engine</label>
                      <select value={modelRight} onChange={(e) => setModelRight(e.target.value)} className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-sm text-slate-200 outline-none">
                        <option>YOLO11s (Attention Enhanced)</option>
                        <option>YOLOv8n (Lightweight)</option>
                        <option>RetinaNet (Baseline)</option>
                      </select>
                    </div>
                  </div>
                )}

                <label className="block text-sm text-slate-400 mb-2">Person Threshold: {pThresh}</label>
                <input type="range" min="0.2" max="0.95" step="0.05" value={pThresh} onChange={(e) => setPThresh(e.target.value)} className="w-full mb-4 accent-blue-500" />
                <label className="block text-sm text-slate-400 mb-2">Gear Threshold: {gThresh}</label>
                <input type="range" min="0.15" max="0.9" step="0.05" value={gThresh} onChange={(e) => setGThresh(e.target.value)} className="w-full mb-6 accent-blue-500" />

                <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-slate-700 border-dashed rounded-xl cursor-pointer hover:border-blue-500 hover:bg-slate-800/50 transition-all">
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <Upload className="w-8 h-8 text-slate-400 mb-2" />
                    <p className="text-sm text-slate-400">Upload Source Image</p>
                  </div>
                  <input type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
                </label>

                {activeTab === 'inspector' ? (
                  <button onClick={runInspection} disabled={!file || loading} className="w-full mt-4 py-3 rounded-xl font-bold bg-blue-600 hover:bg-blue-500 disabled:opacity-50 transition-all shadow-[0_0_15px_rgba(37,99,235,0.4)]">
                    {loading ? "Scanning Frame..." : "Run Inspection"}
                  </button>
                ) : (
                  <button onClick={runDualInspection} disabled={!file || dualLoading} className="w-full mt-4 py-3 rounded-xl font-bold bg-purple-600 hover:bg-purple-500 disabled:opacity-50 transition-all shadow-[0_0_15px_rgba(147,51,234,0.4)]">
                    {dualLoading ? "Running Models..." : "Run Dual Compare"}
                  </button>
                )}
              </div>
            </div>

            {/* Main Output Area */}
            <div className="lg:col-span-3 space-y-6">

              {/* TAB 1: INSPECTOR CONTENT */}
              {activeTab === 'inspector' && (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="md:col-span-2 space-y-4">
                      <div className="bg-slate-900/50 backdrop-blur-md rounded-2xl border border-slate-800 overflow-hidden flex items-center justify-center min-h-[450px]">
                        {resultData ? (
                          <img src={resultData.image} alt="Inference Result" className="w-full h-auto object-contain" />
                        ) : image ? (
                          <img src={image} alt="Original" className="w-full h-auto object-contain opacity-60" />
                        ) : (
                          <p className="text-slate-500 flex flex-col items-center gap-2"><ImageIcon size={32} className="opacity-50" /> Awaiting visual input...</p>
                        )}
                      </div>
                    </div>

                    <div className="space-y-6">
                      <div className="bg-slate-900/50 backdrop-blur-md p-5 rounded-2xl border border-slate-800">
                        <h3 className="text-lg font-semibold mb-4 text-slate-300">Live Telemetry</h3>
                        <div className="grid grid-cols-2 gap-3">
                          <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/50 text-center">
                            <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Total</p>
                            <p className="text-3xl font-bold">{resultData?.telemetry.total || 0}</p>
                          </div>
                          <div className="bg-emerald-900/20 p-3 rounded-lg border border-emerald-800/30 text-center">
                            <p className="text-xs text-emerald-400 uppercase tracking-wider mb-1">Safe</p>
                            <p className="text-3xl font-bold text-emerald-400">{resultData?.telemetry.safe || 0}</p>
                          </div>
                          <div className="bg-red-900/20 p-3 rounded-lg border border-red-800/30 text-center">
                            <p className="text-xs text-red-400 uppercase tracking-wider mb-1">Critical</p>
                            <p className="text-3xl font-bold text-red-400">{resultData?.telemetry.crit || 0}</p>
                          </div>
                        </div>
                        {resultData && (
                          <div className="mt-4 pt-4 border-t border-slate-800 flex justify-between items-center text-sm">
                            <span className="text-slate-400">Inference Speed:</span>
                            <span className="font-mono text-blue-400">{resultData.telemetry.latency_ms} ms</span>
                          </div>
                        )}
                      </div>

                      {resultData?.audit && (
                        <div className="bg-slate-900/50 backdrop-blur-md rounded-2xl border border-slate-800 flex flex-col h-[320px]">
                          <div className="p-4 border-b border-slate-800"><h3 className="font-semibold text-slate-300">Worker Audit</h3></div>
                          <div className="p-4 overflow-y-auto space-y-3 flex-1">
                            {resultData.audit.map((worker, i) => (
                              <div key={i} className={`p-4 rounded-xl border ${worker.status === 'Safe' ? 'bg-emerald-900/10 border-emerald-900/30' : 'bg-red-900/10 border-red-900/30'}`}>
                                <div className="flex justify-between items-center mb-3">
                                  <span className="font-bold text-slate-200">{worker.id}</span>
                                  <span className="text-xs bg-slate-950 px-2 py-1 rounded-md border border-slate-800 font-mono text-slate-400">Conf: {worker.conf}</span>
                                </div>
                                <div className="space-y-1 text-sm">
                                  <div className="flex justify-between"><span className="text-slate-400">Hardhat:</span><span>{worker.helmet}</span></div>
                                  <div className="flex justify-between"><span className="text-slate-400">Safety Vest:</span><span>{worker.vest}</span></div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </>
              )}

              {/* TAB 1.5: DUAL COMPARISON CONTENT */}
              {activeTab === 'dual' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-full">
                  {/* Left Model Output */}
                  <div className="bg-slate-900/50 backdrop-blur-md p-4 rounded-2xl border border-slate-800 flex flex-col">
                    <div className="flex justify-between items-center mb-4 pb-2 border-b border-slate-800">
                      <h3 className="font-bold text-blue-400 truncate pr-2">{modelLeft}</h3>
                      {dualResultLeft && <span className="bg-blue-900/30 text-blue-300 border border-blue-800/50 px-3 py-1 rounded-full text-xs font-mono">{dualResultLeft.telemetry.latency_ms} ms</span>}
                    </div>
                    <div className="flex-1 rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center min-h-[300px]">
                      {dualResultLeft ? (
                        <img src={dualResultLeft.image} alt="Left Result" className="w-full h-auto object-contain" />
                      ) : image ? (
                        <img src={image} alt="Original" className="w-full h-auto object-contain opacity-30" />
                      ) : (
                        <p className="text-slate-600 text-sm">Upload an image to compare</p>
                      )}
                    </div>
                  </div>

                  {/* Right Model Output */}
                  <div className="bg-slate-900/50 backdrop-blur-md p-4 rounded-2xl border border-slate-800 flex flex-col">
                    <div className="flex justify-between items-center mb-4 pb-2 border-b border-slate-800">
                      <h3 className="font-bold text-purple-400 truncate pr-2">{modelRight}</h3>
                      {dualResultRight && <span className="bg-purple-900/30 text-purple-300 border border-purple-800/50 px-3 py-1 rounded-full text-xs font-mono">{dualResultRight.telemetry.latency_ms} ms</span>}
                    </div>
                    <div className="flex-1 rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center min-h-[300px]">
                      {dualResultRight ? (
                        <img src={dualResultRight.image} alt="Right Result" className="w-full h-auto object-contain" />
                      ) : image ? (
                        <img src={image} alt="Original" className="w-full h-auto object-contain opacity-30" />
                      ) : (
                        <p className="text-slate-600 text-sm">Upload an image to compare</p>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: ARCHITECTURE BENCHMARKS */}
        {activeTab === 'benchmark' && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-6">
            <div className="bg-slate-900/50 backdrop-blur-md p-6 rounded-2xl border border-slate-800">
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2"><BarChart3 className="text-blue-500" /> Performance Trade-offs</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-slate-700 text-slate-400 text-sm">
                      <th className="pb-3 px-4">Architecture</th>
                      <th className="pb-3 px-4">Parameters (M)</th>
                      <th className="pb-3 px-4">GFLOPs</th>
                      <th className="pb-3 px-4">Est. FPS (CPU)</th>
                      <th className="pb-3 px-4">mAP@50 Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkData.map((row, idx) => (
                      <tr key={idx} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                        <td className="py-4 px-4 font-semibold">{row.name}</td>
                        <td className="py-4 px-4 font-mono text-slate-300">{row.params}M</td>
                        <td className="py-4 px-4 font-mono text-slate-300">{row.gflops}</td>
                        <td className="py-4 px-4 font-mono text-slate-300">{row.fps}</td>
                        <td className="py-4 px-4">
                          <div className="flex items-center gap-3">
                            <span className="font-mono w-12">{row.map50}%</span>
                            <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden max-w-[200px]"><div className="bg-blue-500 h-full rounded-full" style={{ width: `${row.map50}%` }}></div></div>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: EVALUATION PROOFS */}
        {activeTab === 'evaluation' && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="bg-slate-900/50 backdrop-blur-md p-6 rounded-2xl border border-slate-800">
              <div className="flex justify-between items-end mb-8 border-b border-slate-800 pb-4">
                <div>
                  <h2 className="text-2xl font-bold mb-2 flex items-center gap-2"><ImageIcon className="text-emerald-500" /> Diagnostic Artifacts</h2>
                  <p className="text-slate-400">Visual validation proofs directly from the Kaggle dataset fine-tuning.</p>
                </div>
                <div className="w-64">
                  <label className="block text-xs text-slate-400 mb-1">Select Architecture Proofs</label>
                  <select value={model} onChange={(e) => setModel(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-sm text-slate-200 outline-none focus:border-emerald-500">
                    <option>YOLO11s (Attention Enhanced)</option>
                    <option>YOLOv8n (Lightweight)</option>
                    <option>RetinaNet (Baseline)</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="space-y-3">
                  <h3 className="font-semibold text-slate-300 flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-blue-500"></div> Confusion Matrix
                  </h3>
                  <div className="bg-white rounded-xl overflow-hidden p-2">
                    {/* Using exact path from your screenshot */}
                    <img src={`${getEvalFolder(model)}confusion_matrix.png`} alt="Confusion Matrix" className="w-full h-auto" onError={(e) => e.target.src = "https://via.placeholder.com/600x400?text=Metrics+Missing+in+Public+Folder"} />
                  </div>
                </div>

                <div className="space-y-3">
                  <h3 className="font-semibold text-slate-300 flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-purple-500"></div> Precision-Recall Curve
                  </h3>
                  <div className="bg-white rounded-xl overflow-hidden p-2">
                    {/* Using exact path from your screenshot */}
                    <img src={`${getEvalFolder(model)}BoxPR_curve.png`} alt="PR Curve" className="w-full h-auto" onError={(e) => e.target.src = "https://via.placeholder.com/600x400?text=Metrics+Missing+in+Public+Folder"} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}