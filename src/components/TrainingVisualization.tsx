import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, TrendingUp, Zap, Settings, Download } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';

interface TrainingData {
  epoch: number;
  teacherAcc: number;
  studentAcc: number;
  teacherLoss: number;
  studentLoss: number;
  kdLoss: number;
  ceLoss: number;
  learningRate: number;
}

const TrainingVisualization: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [trainingData, setTrainingData] = useState<TrainingData[]>([]);
  const [hyperparams, setHyperparams] = useState({
    temperature: 4.0,
    alpha: 0.7,
    learningRate: 0.001,
    batchSize: 128
  });
  const [showSettings, setShowSettings] = useState(false);
  
  const maxEpochs = 100;
  const trainingSpeed = 100; // ms per epoch

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isTraining && currentEpoch < maxEpochs) {
      interval = setInterval(() => {
        setCurrentEpoch(prev => {
          const newEpoch = prev + 1;
          
          // Simulate realistic training curves with hyperparameter effects
          const tempEffect = Math.max(0.8, 1.0 - (hyperparams.temperature - 4.0) * 0.05);
          const alphaEffect = Math.max(0.9, hyperparams.alpha);
          
          // Teacher training curve
          const teacherAcc = Math.min(0.935, 0.1 + 0.835 * (1 - Math.exp(-newEpoch / 25)));
          const teacherLoss = Math.max(0.2, 2.0 * Math.exp(-newEpoch / 20));
          
          // Student training curve (affected by hyperparameters)
          const baseStudentAcc = 0.1 + 0.812 * (1 - Math.exp(-newEpoch / 30));
          const studentAcc = Math.min(0.912 * tempEffect * alphaEffect, baseStudentAcc);
          const studentLoss = Math.max(0.25, 2.0 * Math.exp(-newEpoch / 25));
          
          // Knowledge distillation components
          const kdLoss = Math.max(0.15, 1.5 * Math.exp(-newEpoch / 20) * (1 / tempEffect));
          const ceLoss = Math.max(0.3, 2.0 * Math.exp(-newEpoch / 25));
          
          // Learning rate decay
          const learningRate = hyperparams.learningRate * Math.pow(0.95, Math.floor(newEpoch / 10));
          
          const newDataPoint: TrainingData = {
            epoch: newEpoch,
            teacherAcc: teacherAcc * 100,
            studentAcc: studentAcc * 100,
            teacherLoss,
            studentLoss,
            kdLoss,
            ceLoss,
            learningRate
          };
          
          setTrainingData(prev => [...prev, newDataPoint]);
          return newEpoch;
        });
      }, trainingSpeed);
    } else if (currentEpoch >= maxEpochs) {
      setIsTraining(false);
    }
    
    return () => clearInterval(interval);
  }, [isTraining, currentEpoch, maxEpochs, hyperparams, trainingSpeed]);

  const resetTraining = () => {
    setCurrentEpoch(0);
    setIsTraining(false);
    setTrainingData([]);
  };

  const currentData = trainingData[trainingData.length - 1];

  return (
    <div className="space-y-8">
      <motion.div 
        className="bg-gradient-to-r from-orange-900/50 to-red-900/50 rounded-xl p-8 border border-gray-800"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Play className="w-8 h-8 text-orange-400" />
            <div>
              <h2 className="text-3xl font-bold">Advanced Training Simulation</h2>
              <p className="text-gray-300">Interactive knowledge distillation with real-time hyperparameter tuning</p>
            </div>
          </div>
          
          <div className="flex space-x-3">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="flex items-center space-x-2 bg-gray-600 hover:bg-gray-700 px-4 py-2 rounded-lg transition-colors"
            >
              <Settings className="w-4 h-4" />
              <span>Settings</span>
            </button>
            <button
              onClick={() => setIsTraining(!isTraining)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                isTraining 
                  ? 'bg-red-600 hover:bg-red-700' 
                  : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {isTraining ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              <span>{isTraining ? 'Pause' : 'Start'}</span>
            </button>
            <button
              onClick={resetTraining}
              className="flex items-center space-x-2 bg-gray-600 hover:bg-gray-700 px-4 py-2 rounded-lg transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset</span>
            </button>
          </div>
        </div>
      </motion.div>

      {/* Hyperparameter Settings */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          >
            <h3 className="text-xl font-semibold mb-4">Hyperparameter Configuration</h3>
            <div className="grid md:grid-cols-4 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Temperature (T)</label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="0.5"
                  value={hyperparams.temperature}
                  onChange={(e) => setHyperparams(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                  className="w-full"
                  disabled={isTraining}
                />
                <span className="text-sm text-blue-400">{hyperparams.temperature}</span>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Alpha (α)</label>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.1"
                  value={hyperparams.alpha}
                  onChange={(e) => setHyperparams(prev => ({ ...prev, alpha: parseFloat(e.target.value) }))}
                  className="w-full"
                  disabled={isTraining}
                />
                <span className="text-sm text-green-400">{hyperparams.alpha}</span>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Learning Rate</label>
                <input
                  type="range"
                  min="0.0001"
                  max="0.01"
                  step="0.0001"
                  value={hyperparams.learningRate}
                  onChange={(e) => setHyperparams(prev => ({ ...prev, learningRate: parseFloat(e.target.value) }))}
                  className="w-full"
                  disabled={isTraining}
                />
                <span className="text-sm text-orange-400">{hyperparams.learningRate.toFixed(4)}</span>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Batch Size</label>
                <select
                  value={hyperparams.batchSize}
                  onChange={(e) => setHyperparams(prev => ({ ...prev, batchSize: parseInt(e.target.value) }))}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-1 text-white"
                  disabled={isTraining}
                >
                  <option value={32}>32</option>
                  <option value={64}>64</option>
                  <option value={128}>128</option>
                  <option value={256}>256</option>
                </select>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Real-time Metrics Dashboard */}
      <div className="grid lg:grid-cols-4 gap-6">
        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          whileHover={{ scale: 1.02 }}
        >
          <div className="flex items-center space-x-3 mb-4">
            <div className="bg-blue-600 p-2 rounded-lg">
              <TrendingUp className="w-5 h-5 text-white" />
            </div>
            <h3 className="text-lg font-semibold">Teacher Accuracy</h3>
          </div>
          <div className="text-3xl font-bold text-blue-400 mb-2">
            {currentData ? `${currentData.teacherAcc.toFixed(1)}%` : '0.0%'}
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div 
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${currentData ? (currentData.teacherAcc / 93.5) * 100 : 0}%` }}
            ></div>
          </div>
        </motion.div>

        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          whileHover={{ scale: 1.02 }}
        >
          <div className="flex items-center space-x-3 mb-4">
            <div className="bg-orange-600 p-2 rounded-lg">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <h3 className="text-lg font-semibold">Student Accuracy</h3>
          </div>
          <div className="text-3xl font-bold text-orange-400 mb-2">
            {currentData ? `${currentData.studentAcc.toFixed(1)}%` : '0.0%'}
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div 
              className="bg-orange-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${currentData ? (currentData.studentAcc / 91.2) * 100 : 0}%` }}
            ></div>
          </div>
        </motion.div>

        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          whileHover={{ scale: 1.02 }}
        >
          <div className="flex items-center space-x-3 mb-4">
            <div className="bg-purple-600 p-2 rounded-lg">
              <span className="text-white font-bold text-sm">KD</span>
            </div>
            <h3 className="text-lg font-semibold">KD Loss</h3>
          </div>
          <div className="text-3xl font-bold text-purple-400 mb-2">
            {currentData ? currentData.kdLoss.toFixed(3) : '0.000'}
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div 
              className="bg-purple-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${currentData ? Math.max(10, (1 - currentData.kdLoss / 1.5) * 100) : 0}%` }}
            ></div>
          </div>
        </motion.div>

        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          whileHover={{ scale: 1.02 }}
        >
          <div className="flex items-center space-x-3 mb-4">
            <div className="bg-green-600 p-2 rounded-lg">
              <span className="text-white font-bold text-sm">LR</span>
            </div>
            <h3 className="text-lg font-semibold">Learning Rate</h3>
          </div>
          <div className="text-2xl font-bold text-green-400 mb-2">
            {currentData ? currentData.learningRate.toExponential(2) : '0.00e+0'}
          </div>
          <div className="text-sm text-gray-400">
            Epoch {currentEpoch}/{maxEpochs}
          </div>
        </motion.div>
      </div>

      {/* Training Progress */}
      <motion.div 
        className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold">Training Progress</h3>
          <div className="flex space-x-4">
            <span className="text-gray-400">Epoch {currentEpoch}/{maxEpochs}</span>
            <button className="flex items-center space-x-1 text-blue-400 hover:text-blue-300">
              <Download className="w-4 h-4" />
              <span className="text-sm">Export Data</span>
            </button>
          </div>
        </div>
        
        <div className="w-full bg-gray-700 rounded-full h-4 mb-6">
          <div 
            className="bg-gradient-to-r from-blue-500 via-purple-500 to-orange-500 h-4 rounded-full transition-all duration-300 flex items-center justify-end pr-2"
            style={{ width: `${(currentEpoch / maxEpochs) * 100}%` }}
          >
            <span className="text-white text-xs font-semibold">
              {((currentEpoch / maxEpochs) * 100).toFixed(0)}%
            </span>
          </div>
        </div>
        
        <div className="grid md:grid-cols-5 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">
              {currentData ? `${currentData.teacherAcc.toFixed(1)}%` : '0.0%'}
            </div>
            <div className="text-sm text-gray-400">Teacher Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-400">
              {currentData ? `${currentData.studentAcc.toFixed(1)}%` : '0.0%'}
            </div>
            <div className="text-sm text-gray-400">Student Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">
              {currentData ? `${Math.abs(currentData.teacherAcc - currentData.studentAcc).toFixed(1)}%` : '0.0%'}
            </div>
            <div className="text-sm text-gray-400">Accuracy Gap</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-400">56x</div>
            <div className="text-sm text-gray-400">Size Reduction</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-400">7.2x</div>
            <div className="text-sm text-gray-400">Speed Improvement</div>
          </div>
        </div>
      </motion.div>

      {/* Interactive Charts */}
      <div className="grid lg:grid-cols-2 gap-8">
        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <h3 className="text-xl font-semibold mb-4">Accuracy Curves</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="epoch" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" domain={[0, 100]} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="teacherAcc" 
                stroke="#3B82F6" 
                strokeWidth={3}
                name="Teacher"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="studentAcc" 
                stroke="#F59E0B" 
                strokeWidth={3}
                name="Student"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>

        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <h3 className="text-xl font-semibold mb-4">Loss Components</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="epoch" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="kdLoss" 
                stroke="#8B5CF6" 
                strokeWidth={2}
                name="KD Loss"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="ceLoss" 
                stroke="#EF4444" 
                strokeWidth={2}
                name="CE Loss"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="studentLoss" 
                stroke="#10B981" 
                strokeWidth={2}
                name="Total Loss"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Knowledge Transfer Visualization */}
      <motion.div 
        className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h3 className="text-xl font-semibold mb-6">Knowledge Transfer Process</h3>
        
        <div className="flex items-center justify-center space-x-8 py-8">
          {/* Teacher */}
          <motion.div 
            className="text-center"
            whileHover={{ scale: 1.05 }}
          >
            <div className="bg-blue-600 w-24 h-24 rounded-lg flex items-center justify-center mb-3 relative">
              <TrendingUp className="w-12 h-12 text-white" />
              {isTraining && (
                <motion.div
                  className="absolute inset-0 border-2 border-blue-400 rounded-lg"
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ repeat: Infinity, duration: 2 }}
                />
              )}
            </div>
            <div className="text-lg font-semibold">Teacher Model</div>
            <div className="text-sm text-gray-400">ResNet-18</div>
            <div className="text-sm text-blue-400">11.2M params</div>
            <div className="text-xs text-green-400 mt-1">
              {currentData ? `${currentData.teacherAcc.toFixed(1)}% acc` : 'Not trained'}
            </div>
          </motion.div>
          
          {/* Knowledge Flow */}
          <div className="flex flex-col items-center">
            <motion.div
              animate={isTraining ? { x: [0, 10, 0] } : {}}
              transition={{ repeat: Infinity, duration: 1.5 }}
              className="flex items-center space-x-2"
            >
              <div className="w-8 h-1 bg-purple-400 rounded"></div>
              <div className="w-6 h-1 bg-purple-400 rounded"></div>
              <div className="w-4 h-1 bg-purple-400 rounded"></div>
            </motion.div>
            <div className="text-xs text-center text-gray-400 mt-2">
              Soft Targets<br />
              T={hyperparams.temperature}, α={hyperparams.alpha}
            </div>
          </div>
          
          {/* Student */}
          <motion.div 
            className="text-center"
            whileHover={{ scale: 1.05 }}
          >
            <div className="bg-orange-600 w-20 h-20 rounded-lg flex items-center justify-center mb-3 relative">
              <Zap className="w-10 h-10 text-white" />
              {isTraining && (
                <motion.div
                  className="absolute inset-0 border-2 border-orange-400 rounded-lg"
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ repeat: Infinity, duration: 2, delay: 0.5 }}
                />
              )}
            </div>
            <div className="text-lg font-semibold">Student Model</div>
            <div className="text-sm text-gray-400">Custom CNN</div>
            <div className="text-sm text-orange-400">0.2M params</div>
            <div className="text-xs text-green-400 mt-1">
              {currentData ? `${currentData.studentAcc.toFixed(1)}% acc` : 'Training...'}
            </div>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
};

export default TrainingVisualization;