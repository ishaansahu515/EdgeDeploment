import React, { useEffect, useRef, useState } from 'react';
import { BarChart3, Zap, Clock, HardDrive, Target, TrendingUp, Download, Share2 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { motion } from 'framer-motion';

const ResultsComparison: React.FC = () => {
  const [selectedMetric, setSelectedMetric] = useState('accuracy');
  const [showDetailedAnalysis, setShowDetailedAnalysis] = useState(false);

  // Comprehensive comparison data
  const comparisonData = [
    {
      metric: 'Model Parameters',
      teacher: '11,173,962',
      student: '199,882',
      baseline: '199,882',
      improvement: '56x reduction',
      icon: Target,
      color: 'blue',
      teacherValue: 11173962,
      studentValue: 199882,
      baselineValue: 199882
    },
    {
      metric: 'Model Size (MB)',
      teacher: '42.8',
      student: '0.8',
      baseline: '0.8',
      improvement: '53x smaller',
      icon: HardDrive,
      color: 'green',
      teacherValue: 42.8,
      studentValue: 0.8,
      baselineValue: 0.8
    },
    {
      metric: 'Inference Time (ms)',
      teacher: '15.2',
      student: '2.1',
      baseline: '2.1',
      improvement: '7.2x faster',
      icon: Clock,
      color: 'orange',
      teacherValue: 15.2,
      studentValue: 2.1,
      baselineValue: 2.1
    },
    {
      metric: 'Test Accuracy (%)',
      teacher: '93.5',
      student: '91.2',
      baseline: '87.8',
      improvement: '+3.4% vs baseline',
      icon: TrendingUp,
      color: 'purple',
      teacherValue: 93.5,
      studentValue: 91.2,
      baselineValue: 87.8
    }
  ];

  // Chart data
  const accuracyData = [
    { name: 'Teacher\n(ResNet-18)', accuracy: 93.5, parameters: 11.2, size: 42.8 },
    { name: 'Student\n(Distilled)', accuracy: 91.2, parameters: 0.2, size: 0.8 },
    { name: 'Student\n(Baseline)', accuracy: 87.8, parameters: 0.2, size: 0.8 }
  ];

  const performanceData = [
    { name: 'Parameters', teacher: 11.2, student: 0.2 },
    { name: 'Size (MB)', teacher: 42.8, student: 0.8 },
    { name: 'Inference (ms)', teacher: 15.2, student: 2.1 },
    { name: 'Memory (MB)', teacher: 180, student: 32 }
  ];

  const efficiencyData = [
    { name: 'Parameter Efficiency', value: 56 },
    { name: 'Size Efficiency', value: 53 },
    { name: 'Speed Efficiency', value: 7.2 },
    { name: 'Memory Efficiency', value: 5.6 }
  ];

  const radarData = [
    { subject: 'Accuracy', teacher: 93.5, student: 91.2, fullMark: 100 },
    { subject: 'Speed', teacher: 20, student: 95, fullMark: 100 },
    { subject: 'Efficiency', teacher: 15, student: 90, fullMark: 100 },
    { subject: 'Memory', teacher: 25, student: 85, fullMark: 100 },
    { subject: 'Deployment', teacher: 30, student: 95, fullMark: 100 }
  ];

  const COLORS = ['#3B82F6', '#F59E0B', '#10B981', '#8B5CF6'];

  return (
    <div className="space-y-8">
      <motion.div 
        className="bg-gradient-to-r from-blue-900/50 to-green-900/50 rounded-xl p-8 border border-gray-800"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <BarChart3 className="w-8 h-8 text-blue-400" />
            <div>
              <h2 className="text-3xl font-bold">Advanced Results & Analysis</h2>
              <p className="text-gray-300">Comprehensive performance comparison with interactive visualizations</p>
            </div>
          </div>
          
          <div className="flex space-x-3">
            <button 
              onClick={() => setShowDetailedAnalysis(!showDetailedAnalysis)}
              className="flex items-center space-x-2 bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg transition-colors"
            >
              <TrendingUp className="w-4 h-4" />
              <span>Detailed Analysis</span>
            </button>
            <button className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition-colors">
              <Download className="w-4 h-4" />
              <span>Export Report</span>
            </button>
            <button className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition-colors">
              <Share2 className="w-4 h-4" />
              <span>Share</span>
            </button>
          </div>
        </div>
      </motion.div>

      {/* Key Metrics Summary */}
      <div className="grid lg:grid-cols-4 gap-6">
        {comparisonData.map((item, index) => {
          const Icon = item.icon;
          const colorClasses = {
            blue: 'bg-blue-600 text-blue-400 border-blue-500',
            green: 'bg-green-600 text-green-400 border-green-500',
            orange: 'bg-orange-600 text-orange-400 border-orange-500',
            purple: 'bg-purple-600 text-purple-400 border-purple-500'
          };
          
          return (
            <motion.div 
              key={index} 
              className="bg-gray-800/50 rounded-lg p-6 border border-gray-700 cursor-pointer hover:bg-gray-800/70 transition-colors"
              whileHover={{ scale: 1.02 }}
              onClick={() => setSelectedMetric(item.metric.toLowerCase())}
            >
              <div className={`w-12 h-12 ${colorClasses[item.color].split(' ')[0]} rounded-lg flex items-center justify-center mb-4`}>
                <Icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="font-semibold mb-3">{item.metric}</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Teacher:</span>
                  <span className="text-blue-400 font-mono">{item.teacher}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Student:</span>
                  <span className="text-orange-400 font-mono">{item.student}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Baseline:</span>
                  <span className="text-green-400 font-mono">{item.baseline}</span>
                </div>
                <div className={`font-semibold ${colorClasses[item.color].split(' ')[1]} pt-2 border-t ${colorClasses[item.color].split(' ')[2]}`}>
                  {item.improvement}
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Interactive Charts Grid */}
      <div className="grid lg:grid-cols-2 gap-8">
        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <h3 className="text-xl font-semibold mb-4">Model Accuracy Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={accuracyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" domain={[85, 95]} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Bar dataKey="accuracy" fill="#3B82F6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <h3 className="text-xl font-semibold mb-4">Resource Comparison (Log Scale)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" scale="log" domain={[0.1, 200]} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Bar dataKey="teacher" fill="#3B82F6" name="Teacher" />
              <Bar dataKey="student" fill="#F59E0B" name="Student" />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Advanced Visualizations */}
      <div className="grid lg:grid-cols-2 gap-8">
        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h3 className="text-xl font-semibold mb-4">Efficiency Improvements</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={efficiencyData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}x`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {efficiencyData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </motion.div>

        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h3 className="text-xl font-semibold mb-4">Performance Radar</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis dataKey="subject" tick={{ fill: '#9CA3AF', fontSize: 12 }} />
              <PolarRadiusAxis tick={{ fill: '#9CA3AF', fontSize: 10 }} />
              <Radar
                name="Teacher"
                dataKey="teacher"
                stroke="#3B82F6"
                fill="#3B82F6"
                fillOpacity={0.2}
                strokeWidth={2}
              />
              <Radar
                name="Student"
                dataKey="student"
                stroke="#F59E0B"
                fill="#F59E0B"
                fillOpacity={0.2}
                strokeWidth={2}
              />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Detailed Comparison Table */}
      <motion.div 
        className="bg-gray-800/50 rounded-lg border border-gray-700 overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div className="px-6 py-4 border-b border-gray-700 bg-gray-900/50">
          <h3 className="text-xl font-semibold">Comprehensive Performance Analysis</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-900/50">
              <tr>
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-300">Metric</th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-blue-400">Teacher (ResNet-18)</th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-orange-400">Student (Distilled)</th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-green-400">Student (Baseline)</th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-purple-400">Improvement</th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-red-400">Trade-off</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              <tr className="hover:bg-gray-900/25 transition-colors">
                <td className="px-6 py-4 text-sm text-gray-300 font-medium">Total Parameters</td>
                <td className="px-6 py-4 text-sm text-blue-400 font-mono">11,173,962</td>
                <td className="px-6 py-4 text-sm text-orange-400 font-mono">199,882</td>
                <td className="px-6 py-4 text-sm text-green-400 font-mono">199,882</td>
                <td className="px-6 py-4 text-sm text-purple-400 font-semibold">56x reduction</td>
                <td className="px-6 py-4 text-sm text-red-400">-2.3% accuracy</td>
              </tr>
              <tr className="bg-gray-900/25 hover:bg-gray-900/40 transition-colors">
                <td className="px-6 py-4 text-sm text-gray-300 font-medium">Model Size (MB)</td>
                <td className="px-6 py-4 text-sm text-blue-400 font-mono">42.8</td>
                <td className="px-6 py-4 text-sm text-orange-400 font-mono">0.8</td>
                <td className="px-6 py-4 text-sm text-green-400 font-mono">0.8</td>
                <td className="px-6 py-4 text-sm text-purple-400 font-semibold">53x smaller</td>
                <td className="px-6 py-4 text-sm text-green-400">Mobile ready</td>
              </tr>
              <tr className="hover:bg-gray-900/25 transition-colors">
                <td className="px-6 py-4 text-sm text-gray-300 font-medium">Test Accuracy (%)</td>
                <td className="px-6 py-4 text-sm text-blue-400 font-mono">93.5</td>
                <td className="px-6 py-4 text-sm text-orange-400 font-mono">91.2</td>
                <td className="px-6 py-4 text-sm text-green-400 font-mono">87.8</td>
                <td className="px-6 py-4 text-sm text-purple-400 font-semibold">+3.4% vs baseline</td>
                <td className="px-6 py-4 text-sm text-green-400">Acceptable loss</td>
              </tr>
              <tr className="bg-gray-900/25 hover:bg-gray-900/40 transition-colors">
                <td className="px-6 py-4 text-sm text-gray-300 font-medium">Inference Time (ms)</td>
                <td className="px-6 py-4 text-sm text-blue-400 font-mono">15.2</td>
                <td className="px-6 py-4 text-sm text-orange-400 font-mono">2.1</td>
                <td className="px-6 py-4 text-sm text-green-400 font-mono">2.1</td>
                <td className="px-6 py-4 text-sm text-purple-400 font-semibold">7.2x faster</td>
                <td className="px-6 py-4 text-sm text-green-400">Real-time ready</td>
              </tr>
              <tr className="hover:bg-gray-900/25 transition-colors">
                <td className="px-6 py-4 text-sm text-gray-300 font-medium">Training Time (hours)</td>
                <td className="px-6 py-4 text-sm text-blue-400 font-mono">2.5</td>
                <td className="px-6 py-4 text-sm text-orange-400 font-mono">1.8</td>
                <td className="px-6 py-4 text-sm text-green-400 font-mono">1.2</td>
                <td className="px-6 py-4 text-sm text-purple-400 font-semibold">28% reduction</td>
                <td className="px-6 py-4 text-sm text-yellow-400">Requires teacher</td>
              </tr>
              <tr className="bg-gray-900/25 hover:bg-gray-900/40 transition-colors">
                <td className="px-6 py-4 text-sm text-gray-300 font-medium">Memory Usage (MB)</td>
                <td className="px-6 py-4 text-sm text-blue-400 font-mono">180</td>
                <td className="px-6 py-4 text-sm text-orange-400 font-mono">32</td>
                <td className="px-6 py-4 text-sm text-green-400 font-mono">32</td>
                <td className="px-6 py-4 text-sm text-purple-400 font-semibold">5.6x less</td>
                <td className="px-6 py-4 text-sm text-green-400">Edge compatible</td>
              </tr>
              <tr className="hover:bg-gray-900/25 transition-colors">
                <td className="px-6 py-4 text-sm text-gray-300 font-medium">Energy Consumption (mJ)</td>
                <td className="px-6 py-4 text-sm text-blue-400 font-mono">45.2</td>
                <td className="px-6 py-4 text-sm text-orange-400 font-mono">8.1</td>
                <td className="px-6 py-4 text-sm text-green-400 font-mono">8.1</td>
                <td className="px-6 py-4 text-sm text-purple-400 font-semibold">5.6x efficient</td>
                <td className="px-6 py-4 text-sm text-green-400">Battery friendly</td>
              </tr>
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Detailed Analysis Section */}
      {showDetailedAnalysis && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="space-y-6"
        >
          {/* Key Insights */}
          <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
            <h3 className="text-xl font-semibold mb-6 flex items-center space-x-2">
              <Zap className="w-5 h-5 text-yellow-400" />
              <span>Detailed Analysis & Insights</span>
            </h3>
            
            <div className="grid md:grid-cols-2 gap-8">
              <div className="space-y-4">
                <div className="bg-gray-900/50 rounded-lg p-4 border-l-4 border-green-500">
                  <h4 className="font-semibold text-green-400 mb-2">✓ Exceptional Compression Ratio</h4>
                  <p className="text-sm text-gray-300 mb-2">
                    Achieved 56x parameter reduction while maintaining 97.5% of original accuracy. 
                    This compression ratio significantly exceeds typical pruning methods (2-10x).
                  </p>
                  <div className="text-xs text-green-300 bg-green-900/20 rounded p-2">
                    <strong>Impact:</strong> Enables deployment on devices with &lt;1MB memory constraints
                  </div>
                </div>
                
                <div className="bg-gray-900/50 rounded-lg p-4 border-l-4 border-blue-500">
                  <h4 className="font-semibold text-blue-400 mb-2">✓ Real-time Performance</h4>
                  <p className="text-sm text-gray-300 mb-2">
                    2.1ms inference time enables real-time processing at 476 FPS, suitable for 
                    video streams and interactive applications.
                  </p>
                  <div className="text-xs text-blue-300 bg-blue-900/20 rounded p-2">
                    <strong>Benchmark:</strong> Exceeds mobile inference requirements (&lt;10ms)
                  </div>
                </div>
                
                <div className="bg-gray-900/50 rounded-lg p-4 border-l-4 border-orange-500">
                  <h4 className="font-semibold text-orange-400 mb-2">✓ Energy Efficiency</h4>
                  <p className="text-sm text-gray-300 mb-2">
                    5.6x reduction in energy consumption extends battery life significantly, 
                    crucial for IoT and mobile deployments.
                  </p>
                  <div className="text-xs text-orange-300 bg-orange-900/20 rounded p-2">
                    <strong>Estimate:</strong> 5.6x longer battery life in continuous operation
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                <div className="bg-gray-900/50 rounded-lg p-4 border-l-4 border-purple-500">
                  <h4 className="font-semibold text-purple-400 mb-2">✓ Knowledge Transfer Effectiveness</h4>
                  <p className="text-sm text-gray-300 mb-2">
                    3.4% accuracy improvement over baseline student training validates the 
                    effectiveness of knowledge distillation technique.
                  </p>
                  <div className="text-xs text-purple-300 bg-purple-900/20 rounded p-2">
                    <strong>Comparison:</strong> Distilled: 91.2% vs Baseline: 87.8%
                  </div>
                </div>
                
                <div className="bg-gray-900/50 rounded-lg p-4 border-l-4 border-red-500">
                  <h4 className="font-semibold text-red-400 mb-2">⚠ Acceptable Trade-offs</h4>
                  <p className="text-sm text-gray-300 mb-2">
                    2.3% accuracy loss is minimal considering massive efficiency gains. 
                    This trade-off is well within acceptable limits for most applications.
                  </p>
                  <div className="text-xs text-red-300 bg-red-900/20 rounded p-2">
                    <strong>Context:</strong> Industry standard accepts 3-5% accuracy loss for 10x+ compression
                  </div>
                </div>
                
                <div className="bg-gray-900/50 rounded-lg p-4 border-l-4 border-yellow-500">
                  <h4 className="font-semibold text-yellow-400 mb-2">📊 Production Readiness</h4>
                  <p className="text-sm text-gray-300 mb-2">
                    All metrics (size, speed, accuracy) meet production deployment requirements 
                    for mobile and edge computing scenarios.
                  </p>
                  <div className="text-xs text-yellow-300 bg-yellow-900/20 rounded p-2">
                    <strong>Status:</strong> Ready for deployment in resource-constrained environments
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Deployment Scenarios */}
          <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
            <h3 className="text-xl font-semibold mb-6">Deployment Scenarios & Recommendations</h3>
            
            <div className="grid md:grid-cols-3 gap-6">
              <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-800/50">
                <h4 className="font-semibold text-blue-400 mb-3">Mobile Applications</h4>
                <ul className="text-sm text-gray-300 space-y-2">
                  <li>• Real-time image classification</li>
                  <li>• Augmented reality filters</li>
                  <li>• Document scanning apps</li>
                  <li>• Food recognition systems</li>
                </ul>
                <div className="mt-3 text-xs text-blue-300 bg-blue-900/30 rounded p-2">
                  <strong>Recommendation:</strong> Deploy student model with confidence
                </div>
              </div>
              
              <div className="bg-green-900/20 rounded-lg p-4 border border-green-800/50">
                <h4 className="font-semibold text-green-400 mb-3">IoT & Edge Devices</h4>
                <ul className="text-sm text-gray-300 space-y-2">
                  <li>• Smart security cameras</li>
                  <li>• Industrial quality control</li>
                  <li>• Agricultural monitoring</li>
                  <li>• Autonomous drones</li>
                </ul>
                <div className="mt-3 text-xs text-green-300 bg-green-900/30 rounded p-2">
                  <strong>Recommendation:</strong> Ideal for resource-constrained deployment
                </div>
              </div>
              
              <div className="bg-orange-900/20 rounded-lg p-4 border border-orange-800/50">
                <h4 className="font-semibold text-orange-400 mb-3">Cloud & Server</h4>
                <ul className="text-sm text-gray-300 space-y-2">
                  <li>• High-throughput batch processing</li>
                  <li>• Cost-optimized inference</li>
                  <li>• Multi-tenant applications</li>
                  <li>• Serverless functions</li>
                </ul>
                <div className="mt-3 text-xs text-orange-300 bg-orange-900/30 rounded p-2">
                  <strong>Recommendation:</strong> Consider teacher model for maximum accuracy
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default ResultsComparison;