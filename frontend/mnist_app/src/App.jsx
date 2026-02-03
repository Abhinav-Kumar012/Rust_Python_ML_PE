import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Landing from './components/Landing';
import InferencePage from './components/InferencePage';

function App() {
    return (
        <Router>
            <div className="min-h-screen bg-neutral-950 text-white selection:bg-neutral-700">
                <Routes>
                    <Route path="/" element={<Landing />} />
                    <Route
                        path="/rust"
                        element={
                            <InferencePage
                                title="Burn (Rust) Inference"
                                apiEndpoint="http://localhost:8080/predict"
                                themeColor="burn"
                            />
                        }
                    />
                    <Route
                        path="/pytorch"
                        element={
                            <InferencePage
                                title="PyTorch Inference"
                                apiEndpoint="http://localhost:5000/predict"
                                themeColor="pytorch"
                            />
                        }
                    />
                </Routes>
            </div>
        </Router>
    );
}

export default App;
