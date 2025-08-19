import React from 'react';
import { Brain, Layers, Zap, ArrowRight } from 'lucide-react';

const ModelArchitecture: React.FC = () => {
  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-purple-900/50 to-blue-900/50 rounded-xl p-8 border border-gray-800">
        <div className="flex items-center space-x-3 mb-4">
          <Brain className="w-8 h-8 text-purple-400" />
          <h2 className="text-3xl font-bold">Model Architecture Comparison</h2>
        </div>
        <p className="text-gray-300 text-lg">
          Comparing the teacher and student model architectures for CIFAR-10 classification
        </p>
      </div>

      {/* Architecture Comparison */}
      <div className="grid md:grid-cols-2 gap-8">
        {/* Teacher Model */}
        <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center space-x-3 mb-6">
            <div className="bg-blue-600 p-2 rounded-lg">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <h3 className="text-xl font-semibold">Teacher Model (ResNet-18)</h3>
          </div>
          
          <div className="space-y-4">
            <div className="bg-gray-900/50 rounded-lg p-4">
              <h4 className="font-semibold text-blue-400 mb-2">Architecture Details</h4>
              <ul className="space-y-2 text-sm text-gray-300">
                <li>• Input: 32×32×3 (CIFAR-10 images)</li>
                <li>• Conv2d(3, 64, 7×7, stride=2, padding=3)</li>
                <li>• MaxPool2d(3×3, stride=2, padding=1)</li>
                <li>• 4 Residual Blocks (64, 128, 256, 512 channels)</li>
                <li>• Global Average Pooling</li>
                <li>• Fully Connected(512, 10)</li>
              </ul>
            </div>
            
            <div className="bg-gray-900/50 rounded-lg p-4">
              <h4 className="font-semibold text-green-400 mb-2">Model Statistics</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-400">Parameters:</span>
                  <div className="font-semibold text-green-400">11.2M</div>
                </div>
                <div>
                  <span className="text-gray-400">Model Size:</span>
                  <div className="font-semibold text-green-400">42.8 MB</div>
                </div>
                <div>
                  <span className="text-gray-400">Accuracy:</span>
                  <div className="font-semibold text-green-400">93.5%</div>
                </div>
                <div>
                  <span className="text-gray-400">Inference:</span>
                  <div className="font-semibold text-green-400">15.2 ms</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Student Model */}
        <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center space-x-3 mb-6">
            <div className="bg-orange-600 p-2 rounded-lg">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <h3 className="text-xl font-semibold">Student Model (Lightweight CNN)</h3>
          </div>
          
          <div className="space-y-4">
            <div className="bg-gray-900/50 rounded-lg p-4">
              <h4 className="font-semibold text-orange-400 mb-2">Architecture Details</h4>
              <ul className="space-y-2 text-sm text-gray-300">
                <li>• Input: 32×32×3 (CIFAR-10 images)</li>
                <li>• Conv2d(3, 32, 3×3, padding=1) + ReLU + BatchNorm</li>
                <li>• Conv2d(32, 64, 3×3, padding=1) + ReLU + MaxPool</li>
                <li>• Conv2d(64, 128, 3×3, padding=1) + ReLU + MaxPool</li>
                <li>• Global Average Pooling + Dropout(0.5)</li>
                <li>• Fully Connected(128, 10)</li>
              </ul>
            </div>
            
            <div className="bg-gray-900/50 rounded-lg p-4">
              <h4 className="font-semibold text-blue-400 mb-2">Model Statistics</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-400">Parameters:</span>
                  <div className="font-semibold text-blue-400">0.2M</div>
                </div>
                <div>
                  <span className="text-gray-400">Model Size:</span>
                  <div className="font-semibold text-blue-400">0.8 MB</div>
                </div>
                <div>
                  <span className="text-gray-400">Accuracy (Distilled):</span>
                  <div className="font-semibold text-blue-400">91.2%</div>
                </div>
                <div>
                  <span className="text-gray-400">Inference:</span>
                  <div className="font-semibold text-blue-400">2.1 ms</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Knowledge Transfer Visualization */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center space-x-3 mb-6">
          <Layers className="w-6 h-6 text-purple-400" />
          <h3 className="text-xl font-semibold">Knowledge Transfer Process</h3>
        </div>
        
        <div className="flex items-center justify-center space-x-8 py-8">
          {/* Teacher */}
          <div className="text-center">
            <div className="bg-blue-600 w-20 h-20 rounded-lg flex items-center justify-center mb-3">
              <Brain className="w-10 h-10 text-white" />
            </div>
            <div className="text-sm font-semibold">Teacher Model</div>
            <div className="text-xs text-gray-400">ResNet-18</div>
            <div className="text-xs text-blue-400">11.2M params</div>
          </div>
          
          {/* Arrow */}
          <div className="flex flex-col items-center">
            <ArrowRight className="w-8 h-8 text-purple-400 mb-2" />
            <div className="text-xs text-center text-gray-400">
              Soft Targets<br />+ Hard Labels
            </div>
          </div>
          
          {/* Student */}
          <div className="text-center">
            <div className="bg-orange-600 w-16 h-16 rounded-lg flex items-center justify-center mb-3">
              <Zap className="w-8 h-8 text-white" />
            </div>
            <div className="text-sm font-semibold">Student Model</div>
            <div className="text-xs text-gray-400">Custom CNN</div>
            <div className="text-xs text-orange-400">0.2M params</div>
          </div>
        </div>
      </div>

      {/* Performance Comparison Chart */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-xl font-semibold mb-4">Performance Trade-offs</h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-gray-900/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-blue-400 mb-2">56x</div>
            <div className="text-sm text-gray-300">Parameter Reduction</div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-green-400 mb-2">7.2x</div>
            <div className="text-sm text-gray-300">Speed Improvement</div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-orange-400 mb-2">2.3%</div>
            <div className="text-sm text-gray-300">Accuracy Loss</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelArchitecture;