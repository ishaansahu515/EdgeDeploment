import React from 'react';
import { Brain, Github, Download } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-gray-900 border-b border-gray-800">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="bg-blue-600 p-2 rounded-lg">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Knowledge Distillation</h1>
              <p className="text-gray-400 text-sm">Model Compression on CIFAR-10</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <button className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition-colors">
              <Download className="w-4 h-4" />
              <span className="text-sm font-medium">Download Code</span>
            </button>
            <button className="flex items-center space-x-2 bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg transition-colors">
              <Github className="w-4 h-4" />
              <span className="text-sm font-medium">GitHub</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;