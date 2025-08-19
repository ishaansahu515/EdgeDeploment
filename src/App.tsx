import React, { useState } from 'react';
import { BookOpen, Brain, BarChart3, Code2, Play, Download, GitBranch, Settings, Zap } from 'lucide-react';
import Header from './components/Header';
import NavigationTabs from './components/NavigationTabs';
import TheorySection from './components/TheorySection';
import ModelArchitecture from './components/ModelArchitecture';
import CodeImplementation from './components/CodeImplementation';
import TrainingVisualization from './components/TrainingVisualization';
import ResultsComparison from './components/ResultsComparison';
import HyperparameterTuning from './components/HyperparameterTuning';
import InteractiveDemo from './components/InteractiveDemo';
import Footer from './components/Footer';

const tabs = [
  { id: 'theory', label: 'Theory & Concepts', icon: BookOpen },
  { id: 'architecture', label: 'Model Architecture', icon: Brain },
  { id: 'code', label: 'Implementation', icon: Code2 },
  { id: 'training', label: 'Training Simulation', icon: Play },
  { id: 'results', label: 'Results & Analysis', icon: BarChart3 },
  { id: 'hyperparams', label: 'Hyperparameter Tuning', icon: Settings },
  { id: 'demo', label: 'Interactive Demo', icon: Zap },
];

function App() {
  const [activeTab, setActiveTab] = useState('theory');

  const renderContent = () => {
    switch (activeTab) {
      case 'theory':
        return <TheorySection />;
      case 'architecture':
        return <ModelArchitecture />;
      case 'code':
        return <CodeImplementation />;
      case 'training':
        return <TrainingVisualization />;
      case 'results':
        return <ResultsComparison />;
      case 'hyperparams':
        return <HyperparameterTuning />;
      case 'demo':
        return <InteractiveDemo />;
      default:
        return <TheorySection />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <Header />
      <NavigationTabs 
        tabs={tabs}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />
      <main className="container mx-auto px-6 py-8">
        {renderContent()}
      </main>
      <Footer />
    </div>
  );
}

export default App;