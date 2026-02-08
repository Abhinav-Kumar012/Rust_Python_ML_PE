import { useState, useRef } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, ArrowRight, Loader2, Link as LinkIcon, RotateCcw } from 'lucide-react';
import { Link } from 'react-router-dom';
import clsx from 'clsx';

const InferencePage = ({ title, apiEndpoint, themeColor }) => {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setPreview(URL.createObjectURL(selectedFile));
            setResult(null);
            setError(null);
        }
    };

    const handleDragOver = (e) => e.preventDefault();

    const handleDrop = (e) => {
        e.preventDefault();
        const selectedFile = e.dataTransfer.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setPreview(URL.createObjectURL(selectedFile));
            setResult(null);
            setError(null);
        }
    };

    const handleAnalyze = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post(apiEndpoint, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setResult(response.data);
        } catch (err) {
            setError(err.message || "Failed to connect to backend");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const reset = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
    };

    return (
        <div className="flex min-h-screen items-center justify-center p-4">
            {/* Background accent */}
            <div className={clsx(
                "fixed inset-0 opacity-10 blur-3xl",
                themeColor === 'burn' ? "bg-orange-900" : "bg-red-900"
            )} />

            <div className="relative w-full max-w-2xl">
                <Link to="/" className="mb-6 inline-flex items-center text-sm text-neutral-400 hover:text-white transition-colors">
                    <ArrowRight className="mr-2 h-4 w-4 rotate-180" /> Back to selection
                </Link>

                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="overflow-hidden rounded-3xl bg-neutral-900/80 backdrop-blur-xl border border-white/10 shadow-2xl"
                >
                    <div className="border-b border-white/10 p-6">
                        <h2 className="text-2xl font-bold">{title}</h2>
                        <p className="text-xs font-mono text-neutral-500 mt-1">{apiEndpoint}</p>
                    </div>

                    <div className="p-8">
                        <div className="grid gap-8 md:grid-cols-2">

                            {/* Left Column: Upload */}
                            <div className="flex flex-col gap-4">
                                <div
                                    className={clsx(
                                        "relative flex aspect-square flex-col items-center justify-center rounded-2xl border-2 border-dashed transition-all",
                                        file ? "border-green-500/50 bg-green-500/5" : "border-neutral-700 hover:border-neutral-500 hover:bg-neutral-800/50 cursor-pointer"
                                    )}
                                    onDragOver={handleDragOver}
                                    onDrop={handleDrop}
                                    onClick={() => !file && fileInputRef.current?.click()}
                                >
                                    {preview ? (
                                        <img src={preview} alt="Preview" className="h-full w-full object-contain p-4" />
                                    ) : (
                                        <div className="flex flex-col items-center p-4 text-center">
                                            <Upload className="mb-4 h-10 w-10 text-neutral-500" />
                                            <p className="text-sm text-neutral-400 mb-2">Drag image here</p>
                                            <button
                                                type="button"
                                                className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 rounded-lg text-sm font-medium text-white transition-colors"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    fileInputRef.current?.click();
                                                }}
                                            >
                                                Select Image
                                            </button>
                                            <p className="mt-4 text-xs text-neutral-600">Supports PNG, JPG</p>
                                        </div>
                                    )}
                                    <input
                                        type="file"
                                        ref={fileInputRef}
                                        className="hidden"
                                        onChange={handleFileChange}
                                        accept="image/*"
                                    />

                                    {file && (
                                        <button
                                            onClick={(e) => { e.stopPropagation(); reset(); }}
                                            className="absolute right-2 top-2 rounded-full bg-neutral-900 p-1.5 text-neutral-400 hover:text-white"
                                        >
                                            <RotateCcw className="w-4 h-4" />
                                        </button>
                                    )}
                                </div>

                                <button
                                    onClick={handleAnalyze}
                                    disabled={!file || loading}
                                    className={clsx(
                                        "flex w-full items-center justify-center rounded-xl py-3 font-semibold transition-all",
                                        !file || loading
                                            ? "bg-neutral-800 text-neutral-500 cursor-not-allowed"
                                            : themeColor === 'burn'
                                                ? "bg-orange-600 hover:bg-orange-500 text-white shadow-lg shadow-orange-900/20"
                                                : "bg-red-600 hover:bg-red-500 text-white shadow-lg shadow-red-900/20"
                                    )}
                                >
                                    {loading ? <Loader2 className="animate-spin mr-2 h-5 w-5" /> : null}
                                    {loading ? 'Processing...' : 'Analyze Digit'}
                                </button>
                            </div>

                            {/* Right Column: Results */}
                            <div className="flex flex-col justify-center rounded-2xl bg-neutral-950/50 p-6 ring-1 ring-white/5">
                                <AnimatePresence mode="wait">
                                    {result ? (
                                        <motion.div
                                            key="result"
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            className="flex flex-col items-center text-center"
                                        >
                                            <p className="text-sm font-medium text-neutral-500 uppercase tracking-widest">Prediction</p>
                                            <div className={clsx(
                                                "mt-4 text-8xl font-black tabular-nums bg-gradient-to-b bg-clip-text text-transparent pb-2",
                                                themeColor === 'burn' ? "from-white to-orange-400" : "from-white to-red-400"
                                            )}>
                                                {result.prediction}
                                            </div>

                                            <div className="mt-8 w-full space-y-3 border-t border-white/5 pt-6 text-left">
                                                {result.meta && (
                                                    <>
                                                        {result.meta.latency_ms && (
                                                            <div className="flex justify-between text-sm">
                                                                <span className="text-neutral-500">Latency</span>
                                                                <span className="font-mono text-white">{result.meta.latency_ms.toFixed(2)}ms</span>
                                                            </div>
                                                        )}
                                                        {result.meta.security_context && (
                                                            <div className="flex justify-between text-sm">
                                                                <span className="text-neutral-500">Service Ver</span>
                                                                <span className="font-mono text-white text-xs">{result.meta.security_context.service_version}</span>
                                                            </div>
                                                        )}
                                                        {result.meta.resources && (
                                                            <div className="flex justify-between text-sm">
                                                                <span className="text-neutral-500">Memory</span>
                                                                <span className="font-mono text-white">{result.meta.resources.available_memory_mb} MB Avail</span>
                                                            </div>
                                                        )}
                                                    </>
                                                )}

                                                {/* Fallback metrics for simple responses */}
                                                {result.latency_ms && !result.meta && (
                                                    <div className="flex justify-between text-sm">
                                                        <span className="text-neutral-500">Latency</span>
                                                        <span className="font-mono text-white">{result.latency_ms.toFixed(2)}ms</span>
                                                    </div>
                                                )}
                                            </div>
                                        </motion.div>
                                    ) : error ? (
                                        <motion.div
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            className="text-center"
                                        >
                                            <div className="bg-red-500/10 text-red-400 p-4 rounded-xl text-sm border border-red-500/20">
                                                {error}
                                            </div>
                                        </motion.div>
                                    ) : (
                                        <div className="flex h-full flex-col items-center justify-center text-center opacity-30">
                                            <BrainCircuit className="h-16 w-16 mb-4" />
                                            <p className="text-sm">Results will appear here</p>
                                        </div>
                                    )}
                                </AnimatePresence>
                            </div>

                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
};

export default InferencePage;
