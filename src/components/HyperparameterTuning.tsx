import React, { useState, useEffect } from 'react';
import { Settings, TrendingUp, Zap, Target, RotateCcw, Play } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar } from 'recharts';
import { motion } from 'framer-motion';

interface HyperparamConfig {
  temperature: number;
  alpha: number;
  learningRate: number;
  batchSize: number;
  weightDecay: number;
}

interface ExperimentResult {
  id: string;
  config: HyperparamConfig;
  finalAccuracy: number;
  convergenceEpoch: number;
  trainingTime: number;
  stability: number;
}

const HyperparameterTuning: React.FC = () => {
  const [currentConfig, setCurrentConfig] = useState<HyperparamConfig>({
    temperature: 4.0,
    alpha: 0.7,
    learningRate: 0.001,
    batchSize: 128,
    weightDecay: 0.0001
  });

  const [experiments, setExperiments] = useState<ExperimentResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState('accuracy');
  const [bestConfig, setBestConfig] = useState<ExperimentResult | null>(null);

  // Predefined experiment results for demonstration
  const predefinedExperiments: ExperimentResult[] = [
    {
      id: 'exp1',
      config: { temperature: 3.0, alpha: 0.6, learningRate: 0.001, batchSize: 64, weightDecay: 0.0001 },
      finalAccuracy: 89.8,
      convergenceEpoch: 45,
      trainingTime: 1.2,
      stability: 0.92
    },
    {
      id: 'exp2',
      config: { temperature: 4.0, alpha: 0.7, learningRate: 0.001, batchSize: 128, weightDecay: 0.0001 },
      finalAccuracy: 91.2,
      convergenceEpoch: 38,
      trainingTime: 1.8,
      stability: 0.95
    },
    {
      id: 'exp3',
      config: { temperature: 5.0, alpha: 0.8, learningRate: 0.0005, batchSize: 256, weightDecay: 0.0005 },
      finalAccuracy: 90.5,
      convergenceEpoch: 52,
      trainingTime: 2.1,
      stability: 0.88
    },
    {
      id: 'exp4',
      config: { temperature: 2.0, alpha: 0.5, learningRate: 0.002, batchSize: 64, weightDecay: 0.0001 },
      finalAccuracy: 88.9,
      convergenceEpoch: 41,
      trainingTime: 1.1,
      stability: 0.85
    },
    {
      id: 'exp5',
      config: { temperature: 6.0, alpha: 0.9, learningRate: 0.0008, batchSize: 128, weightDecay: 0.0002 },
      finalAccuracy: 89.7,
      convergenceEpoch: 48,
      trainingTime: 1.9,
      stability: 0.90
    }
  ];

  useEffect(() => {
    setExperiments(predefinedExperiments);
    setBestConfig(predefinedExperiments.reduce((best, current) => 
      current.finalAccuracy > best.finalAccuracy ? current : best
    ));
  }, []);

  const runExperiment = () => {
    setIsRunning(true);
    
    // Simulate experiment running
    setTimeout(() => {
      const newExperiment: ExperimentResult = {
        id: `exp${experiments.length + 1}`,
        config: { ...currentConfig },
        finalAccuracy: 87 + Math.random() * 5, // Simulate accuracy between 87-92%
        convergenceEpoch: 30 + Math.floor(Math.random() * 30),
        trainingTime: 1.0 + Math.random() * 1.5,
        stability: 0.8 + Math.random() * 0.2
      };
      
      setExperiments(prev => [...prev, newExperiment]);
      
      if (!bestConfig || newExperiment.finalAccuracy > bestConfig.finalAccuracy) {
        setBestConfig(newExperiment);
      }
      
      setIsRunning(false);
    }, 3000);
  };

  const resetExperiments = () => {
    setExperiments(predefinedExperiments);
    setBestConfig(predefinedExperiments.reduce((best, current) => 
      current.finalAccuracy > best.finalAccuracy ? current : best
    ));
  };

  // Prepare data for visualizations
  const scatterData = experiments.map(exp => ({
    temperature: exp.config.temperature,
    alpha: exp.config.alpha,
    accuracy: exp.finalAccuracy,
    learningRate: exp.config.learningRate,
    batchSize: exp.config.batchSize,
    id: exp.id
  }));

  const parameterImportance = [
    { parameter: 'Temperature', importance: 0.35, impact: 'High' },
    { parameter: 'Alpha', importance: 0.28, impact: 'High' },
    { parameter: 'Learning Rate', importance: 0.22, impact: 'Medium' },
    { parameter: 'Batch Size', importance: 0.10, impact: 'Low' },
    { parameter: 'Weight Decay', importance: 0.05, impact: 'Low' }
  ];

  return (
    <div className="space-y-8">
      <motion.div 
        className="bg-gradient-to-r from-purple-900/50 to-pink-900/50 rounded-xl p-8 border border-gray-800"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Settings className="w-8 h-8 text-purple-400" />
            <div>
              <h2 className="text-3xl font-bold">Hyperparameter Optimization</h2>
              <p className="text-gray-300">Systematic tuning for optimal knowledge distillation performance</p>
            </div>
          </div>
          
          <div className="flex space-x-3">
            <button
              onClick={resetExperiments}
              className="flex items-center space-x-2 bg-gray-600 hover:bg-gray-700 px-4 py-2 rounded-lg transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset</span>
            </button>
            <button
              onClick={runExperiment}
              disabled={isRunning}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                isRunning 
                  ? 'bg-gray-600 cursor-not-allowed' 
                  : 'bg-purple-600 hover:bg-purple-700'
              }`}
            >
              <Play className="w-4 h-4" />
              <span>{isRunning ? 'Running...' : 'Run Experiment'}</span>
            </button>
          </div>
        </div>
      </motion.div>

      {/* Current Configuration */}
      <div className="grid lg:grid-cols-2 gap-8">
        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <h3 className="text-xl font-semibold mb-6">Current Configuration</h3>
          
          <div className="space-y-6">
            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm font-medium text-gray-300">Temperature (T)</label>
                <span className="text-blue-400 font-mono">{currentConfig.temperature}</span>
              </div>
              <input
                type="range"
                min="1"
                max="10"
                step="0.5"
                value={currentConfig.temperature}
                onChange={(e) => setCurrentConfig(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                disabled={isRunning}
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>1.0</span>
                <span>10.0</span>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm font-medium text-gray-300">Alpha (α)</label>
                <span className="text-green-400 font-mono">{currentConfig.alpha}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.1"
                value={currentConfig.alpha}
                onChange={(e) => setCurrentConfig(prev => ({ ...prev, alpha: parseFloat(e.target.value) }))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                disabled={isRunning}
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>0.1</span>
                <span>0.9</span>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm font-medium text-gray-300">Learning Rate</label>
                <span className="text-orange-400 font-mono">{currentConfig.learningRate.toFixed(4)}</span>
              </div>
              <input
                type="range"
                min="0.0001"
                max="0.01"
                step="0.0001"
                value={currentConfig.learningRate}
                onChange={(e) => setCurrentConfig(prev => ({ ...prev, learningRate: parseFloat(e.target.value) }))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                disabled={isRunning}
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>0.0001</span>
                <span>0.01</span>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Batch Size</label>
                <select
                  value={currentConfig.batchSize}
                  onChange={(e) => setCurrentConfig(prev => ({ ...prev, batchSize: parseInt(e.target.value) }))}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
                  disabled={isRunning}
                >
                  <option value={32}>32</option>
                  <option value={64}>64</option>
                  <option value={128}>128</option>
                  <option value={256}>256</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Weight Decay</label>
                <select
                  value={currentConfig.weightDecay}
                  onChange={(e) => setCurrentConfig(prev => ({ ...prev, weightDecay: parseFloat(e.target.value) }))}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
                  disabled={isRunning}
                >
                  <option value={0.0001}>0.0001</option>
                  <option value={0.0005}>0.0005</option>
                  <option value={0.001}>0.001</option>
                  <option value={0.005}>0.005</option>
                </select>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Best Configuration */}
        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <h3 className="text-xl font-semibold mb-6 flex items-center space-x-2">
            <Target className="w-5 h-5 text-yellow-400" />
            <span>Best Configuration Found</span>
          </h3>
          
          {bestConfig && (
            <div className="space-y-4">
              <div className="bg-yellow-900/20 rounded-lg p-4 border border-yellow-800/50">
                <div className="text-2xl font-bold text-yellow-400 mb-2">
                  {bestConfig.finalAccuracy.toFixed(1)}% Accuracy
                </div>
                <div className="text-sm text-gray-300">
                  Converged in {bestConfig.convergenceEpoch} epochs
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-900/50 rounded-lg p-3">
                  <div className="text-sm text-gray-400">Temperature</div>
                  <div className="text-lg font-semibold text-blue-400">{bestConfig.config.temperature}</div>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-3">
                  <div className="text-sm text-gray-400">Alpha</div>
                  <div className="text-lg font-semibold text-green-400">{bestConfig.config.alpha}</div>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-3">
                  <div className="text-sm text-gray-400">Learning Rate</div>
                  <div className="text-lg font-semibold text-orange-400">{bestConfig.config.learningRate}</div>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-3">
                  <div className="text-sm text-gray-400">Batch Size</div>
                  <div className="text-lg font-semibold text-purple-400">{bestConfig.config.batchSize}</div>
                </div>
              </div>
              
              <button
                onClick={() => setCurrentConfig(bestConfig.config)}
                className="w-full bg-yellow-600 hover:bg-yellow-700 text-white py-2 px-4 rounded-lg transition-colors"
                disabled={isRunning}
              >
                Apply Best Configuration
              </button>
            </div>
          )}
        </motion.div>
      </div>

      {/* Experiment Results */}
      <motion.div 
        className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold">Experiment Results</h3>
          <div className="flex space-x-2">
            <button
              onClick={() => setSelectedMetric('accuracy')}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                selectedMetric === 'accuracy' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Accuracy
            </button>
            <button
              onClick={() => setSelectedMetric('convergence')}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                selectedMetric === 'convergence' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Convergence
            </button>
            <button
              onClick={() => setSelectedMetric('stability')}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                selectedMetric === 'stability' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Stability
            </button>
          </div>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-900/50">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-300">Experiment</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-300">Temperature</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-300">Alpha</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-300">LR</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-300">Batch Size</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-300">Accuracy</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-300">Convergence</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-300">Stability</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {experiments.map((exp, index) => (
                <tr 
                  key={exp.id} 
                  className={`hover:bg-gray-900/25 transition-colors ${
                    bestConfig?.id === exp.id ? 'bg-yellow-900/20 border-l-4 border-yellow-500' : ''
                  }`}
                >
                  <td className="px-4 py-3 text-sm text-gray-300 font-mono">{exp.id}</td>
                  <td className="px-4 py-3 text-sm text-blue-400 font-mono">{exp.config.temperature}</td>
                  <td className="px-4 py-3 text-sm text-green-400 font-mono">{exp.config.alpha}</td>
                  <td className="px-4 py-3 text-sm text-orange-400 font-mono">{exp.config.learningRate}</td>
                  <td className="px-4 py-3 text-sm text-purple-400 font-mono">{exp.config.batchSize}</td>
                  <td className="px-4 py-3 text-sm text-white font-semibold">{exp.finalAccuracy.toFixed(1)}%</td>
                  <td className="px-4 py-3 text-sm text-gray-300">{exp.convergenceEpoch} epochs</td>
                  <td className="px-4 py-3 text-sm text-gray-300">{(exp.stability * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Visualization Charts */}
      <div className="grid lg:grid-cols-2 gap-8">
        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <h3 className="text-xl font-semibold mb-4">Temperature vs Alpha Impact</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={scatterData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="temperature" stroke="#9CA3AF" name="Temperature" />
              <YAxis dataKey="alpha" stroke="#9CA3AF" name="Alpha" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
                formatter={(value, name) => [
                  name === 'accuracy' ? `${value.toFixed(1)}%` : value,
                  name === 'accuracy' ? 'Accuracy' : name
                ]}
              />
              <Scatter dataKey="accuracy" fill="#3B82F6" />
            </ScatterChart>
          </ResponsiveContainer>
        </motion.div>

        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <h3 className="text-xl font-semibold mb-4">Parameter Importance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={parameterImportance} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis type="number" stroke="#9CA3AF" />
              <YAxis dataKey="parameter" type="category" stroke="#9CA3AF" width={100} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
                formatter={(value) => [`${(value * 100).toFixed(0)}%`, 'Importance']}
              />
              <Bar dataKey="importance" fill="#F59E0B" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Hyperparameter Guidelines */}
      <motion.div 
        className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h3 className="text-xl font-semibold mb-6">Hyperparameter Tuning Guidelines</h3>
        
        <div className="grid md:grid-cols-2 gap-8">
          <div className="space-y-4">
            <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-800/50">
              <h4 className="font-semibold text-blue-400 mb-2">Temperature (T)</h4>
              <p className="text-sm text-gray-300 mb-2">
                Controls the softness of probability distributions. Higher values create softer distributions.
              </p>
              <div className="text-xs text-blue-300 space-y-1">
                <div><strong>Optimal Range:</strong> 3.0 - 5.0</div>
                <div><strong>Too Low (&lt;2):</strong> Hard targets, limited knowledge transfer</div>
                <div><strong>Too High (&gt;8):</strong> Over-smoothed distributions, poor convergence</div>
              </div>
            </div>
            
            <div className="bg-green-900/20 rounded-lg p-4 border border-green-800/50">
              <h4 className="font-semibold text-green-400 mb-2">Alpha (α)</h4>
              <p className="text-sm text-gray-300 mb-2">
                Balances knowledge distillation loss and cross-entropy loss. Higher values favor distillation.
              </p>
              <div className="text-xs text-green-300 space-y-1">
                <div><strong>Optimal Range:</strong> 0.6 - 0.8</div>
                <div><strong>Too Low (&lt;0.5):</strong> Insufficient knowledge transfer</div>
                <div><strong>Too High (&gt;0.9):</strong> Ignores ground truth labels</div>
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="bg-orange-900/20 rounded-lg p-4 border border-orange-800/50">
              <h4 className="font-semibold text-orange-400 mb-2">Learning Rate</h4>
              <p className="text-sm text-gray-300 mb-2">
                Controls the step size during optimization. Affects convergence speed and stability.
              </p>
              <div className="text-xs text-orange-300 space-y-1">
                <div><strong>Optimal Range:</strong> 0.0005 - 0.002</div>
                <div><strong>Too Low (&lt;0.0001):</strong> Slow convergence</div>
                <div><strong>Too High (&gt;0.005):</strong> Unstable training, oscillations</div>
              </div>
            </div>
            
            <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-800/50">
              <h4 className="font-semibold text-purple-400 mb-2">Batch Size</h4>
              <p className="text-sm text-gray-300 mb-2">
                Number of samples processed together. Affects gradient estimation and memory usage.
              </p>
              <div className="text-xs text-purple-300 space-y-1">
                <div><strong>Optimal Range:</strong> 64 - 256</div>
                <div><strong>Too Small (&lt;32):</strong> Noisy gradients, unstable training</div>
                <div><strong>Too Large (&gt;512):</strong> Memory constraints, poor generalization</div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default HyperparameterTuning;