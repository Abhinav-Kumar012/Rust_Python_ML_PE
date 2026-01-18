import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Flame, BrainCircuit } from 'lucide-react';

const Card = ({ to, title, icon: Icon, color, description }) => {
    return (
        <Link to={to} className="group relative block w-full max-w-sm">
            <div className={`absolute -inset-0.5 rounded-2xl bg-gradient-to-r ${color} opacity-75 blur transition duration-500 group-hover:opacity-100`}></div>
            <div className="relative flex h-full flex-col items-center justify-center rounded-2xl bg-neutral-900 p-8 text-center sm:p-12">
                <div className="mb-6 rounded-full bg-neutral-800 p-4 ring-1 ring-white/10 transition-transform duration-300 group-hover:scale-110 group-hover:ring-white/20">
                    <Icon className="h-12 w-12 text-white/90" />
                </div>
                <h3 className="mb-2 text-2xl font-bold text-white">{title}</h3>
                <p className="text-sm text-neutral-400">{description}</p>
            </div>
        </Link>
    );
};

const Landing = () => {
    return (
        <div className="flex min-h-screen flex-col items-center justify-center px-4 py-12">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-16 text-center"
            >
                <h1 className="mb-4 text-4xl font-extrabold tracking-tight text-white sm:text-6xl">
                    MNIST Inference Engine
                </h1>
                <p className="mx-auto max-w-2xl text-lg text-neutral-400">
                    Select your inference backend to classify handwritten digits.
                </p>
            </motion.div>

            <div className="grid w-full max-w-4xl gap-8 sm:grid-cols-2 lg:gap-12">
                <motion.div
                    initial={{ opacity: 0, x: -50 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 }}
                    className="flex justify-center"
                >
                    <Card
                        to="/rust"
                        title="Burn (Rust)"
                        icon={Flame}
                        color="from-orange-600 to-red-600"
                        description="High-performance CPU inference via Burn/NdArray backend."
                    />
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, x: 50 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="flex justify-center"
                >
                    <Card
                        to="/pytorch"
                        title="PyTorch"
                        icon={BrainCircuit}
                        color="from-red-500 to-orange-400"
                        description="Standard inference via ONNX Runtime / PyTorch backend."
                    />
                </motion.div>
            </div>

            <div className="fixed bottom-8 text-xs text-neutral-600">
                Rust + Python ML Architecture Demo
            </div>
        </div>
    );
};

export default Landing;
