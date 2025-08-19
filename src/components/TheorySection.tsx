import React from 'react';
import { BookOpen, ArrowRight, Lightbulb, Target, Zap, Brain, TrendingUp } from 'lucide-react';
import { motion } from 'framer-motion';

const TheorySection: React.FC = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5
      }
    }
  };

  return (
    <motion.div 
      className="space-y-8"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Hero Section */}
      <motion.div 
        className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 rounded-xl p-8 border border-gray-800"
        variants={itemVariants}
      >
        <div className="flex items-center space-x-3 mb-4">
          <BookOpen className="w-8 h-8 text-blue-400" />
          <h2 className="text-3xl font-bold">Knowledge Distillation Theory</h2>
        </div>
        <p className="text-gray-300 text-lg leading-relaxed mb-6">
          Knowledge Distillation is a model compression technique that transfers the learned knowledge 
          from a large, complex "teacher" model to a smaller, efficient "student" model while preserving 
          most of the original accuracy.
        </p>
        
        {/* Key Benefits */}
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-blue-900/30 rounded-lg p-4 border border-blue-800/50">
            <Brain className="w-6 h-6 text-blue-400 mb-2" />
            <h4 className="font-semibold text-blue-300 mb-1">Model Compression</h4>
            <p className="text-sm text-gray-300">Reduce model size by 10-100x while maintaining performance</p>
          </div>
          <div className="bg-green-900/30 rounded-lg p-4 border border-green-800/50">
            <Zap className="w-6 h-6 text-green-400 mb-2" />
            <h4 className="font-semibold text-green-300 mb-1">Faster Inference</h4>
            <p className="text-sm text-gray-300">Achieve 5-10x speedup for real-time applications</p>
          </div>
          <div className="bg-purple-900/30 rounded-lg p-4 border border-purple-800/50">
            <TrendingUp className="w-6 h-6 text-purple-400 mb-2" />
            <h4 className="font-semibold text-purple-300 mb-1">Edge Deployment</h4>
            <p className="text-sm text-gray-300">Deploy on mobile and IoT devices efficiently</p>
          </div>
        </div>
      </motion.div>

      {/* Problem Statement */}
      <div className="grid md:grid-cols-2 gap-8">
        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          variants={itemVariants}
        >
          <div className="flex items-center space-x-3 mb-4">
            <Target className="w-6 h-6 text-orange-400" />
            <h3 className="text-xl font-semibold">Problem Statement</h3>
          </div>
          <ul className="space-y-3 text-gray-300">
            <li className="flex items-start space-x-2">
              <ArrowRight className="w-4 h-4 mt-0.5 text-orange-400 flex-shrink-0" />
              <span>Large models achieve high accuracy but require significant computational resources</span>
            </li>
            <li className="flex items-start space-x-2">
              <ArrowRight className="w-4 h-4 mt-0.5 text-orange-400 flex-shrink-0" />
              <span>Edge devices have limited processing power and memory constraints</span>
            </li>
            <li className="flex items-start space-x-2">
              <ArrowRight className="w-4 h-4 mt-0.5 text-orange-400 flex-shrink-0" />
              <span>Need for efficient models without significant accuracy degradation</span>
            </li>
            <li className="flex items-start space-x-2">
              <ArrowRight className="w-4 h-4 mt-0.5 text-orange-400 flex-shrink-0" />
              <span>Real-time inference requirements for production systems</span>
            </li>
          </ul>
        </motion.div>

        <motion.div 
          className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
          variants={itemVariants}
        >
          <div className="flex items-center space-x-3 mb-4">
            <Lightbulb className="w-6 h-6 text-green-400" />
            <h3 className="text-xl font-semibold">Solution Approach</h3>
          </div>
          <ul className="space-y-3 text-gray-300">
            <li className="flex items-start space-x-2">
              <ArrowRight className="w-4 h-4 mt-0.5 text-green-400 flex-shrink-0" />
              <span>Train a large, accurate teacher model (ResNet-18/34) on CIFAR-10</span>
            </li>
            <li className="flex items-start space-x-2">
              <ArrowRight className="w-4 h-4 mt-0.5 text-green-400 flex-shrink-0" />
              <span>Design a lightweight student model with fewer parameters</span>
            </li>
            <li className="flex items-start space-x-2">
              <ArrowRight className="w-4 h-4 mt-0.5 text-green-400 flex-shrink-0" />
              <span>Use knowledge distillation to transfer learned representations</span>
            </li>
            <li className="flex items-start space-x-2">
              <ArrowRight className="w-4 h-4 mt-0.5 text-green-400 flex-shrink-0" />
              <span>Optimize hyperparameters for best compression-accuracy trade-off</span>
            </li>
          </ul>
        </motion.div>
      </div>

      {/* Knowledge Distillation Process */}
      <motion.div 
        className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
        variants={itemVariants}
      >
        <h3 className="text-xl font-semibold mb-6 flex items-center space-x-2">
          <Brain className="w-5 h-5 text-purple-400" />
          <span>Knowledge Distillation Process</span>
        </h3>
        
        <div className="grid md:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="bg-blue-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-white font-bold">1</span>
            </div>
            <h4 className="font-semibold text-blue-400 mb-2">Teacher Training</h4>
            <p className="text-sm text-gray-300">Train large model to high accuracy on dataset</p>
          </div>
          
          <div className="text-center">
            <div className="bg-green-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-white font-bold">2</span>
            </div>
            <h4 className="font-semibold text-green-400 mb-2">Soft Target Generation</h4>
            <p className="text-sm text-gray-300">Generate probability distributions with temperature scaling</p>
          </div>
          
          <div className="text-center">
            <div className="bg-orange-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-white font-bold">3</span>
            </div>
            <h4 className="font-semibold text-orange-400 mb-2">Student Training</h4>
            <p className="text-sm text-gray-300">Train compact model using combined loss function</p>
          </div>
          
          <div className="text-center">
            <div className="bg-purple-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-white font-bold">4</span>
            </div>
            <h4 className="font-semibold text-purple-400 mb-2">Evaluation</h4>
            <p className="text-sm text-gray-300">Compare performance metrics and deployment readiness</p>
          </div>
        </div>
      </motion.div>

      {/* Key Concepts */}
      <motion.div 
        className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
        variants={itemVariants}
      >
        <h3 className="text-xl font-semibold mb-6 flex items-center space-x-2">
          <Zap className="w-5 h-5 text-blue-400" />
          <span>Key Concepts & Techniques</span>
        </h3>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="bg-gray-900/50 rounded-lg p-4 border border-blue-800/30">
            <h4 className="font-semibold text-blue-400 mb-3">Soft Targets</h4>
            <p className="text-sm text-gray-300 mb-3">
              Probability distributions from teacher model contain richer information than hard labels, 
              providing insights into class similarities and decision boundaries.
            </p>
            <div className="bg-blue-900/20 rounded p-2 text-xs font-mono text-blue-300">
              softmax(logits / T)
            </div>
          </div>
          
          <div className="bg-gray-900/50 rounded-lg p-4 border border-green-800/30">
            <h4 className="font-semibold text-green-400 mb-3">Temperature Scaling</h4>
            <p className="text-sm text-gray-300 mb-3">
              Temperature parameter T controls the "softness" of probability distributions. 
              Higher T creates softer distributions with more information.
            </p>
            <div className="bg-green-900/20 rounded p-2 text-xs font-mono text-green-300">
              T ∈ [1, 20], optimal ≈ 3-5
            </div>
          </div>
          
          <div className="bg-gray-900/50 rounded-lg p-4 border border-orange-800/30">
            <h4 className="font-semibold text-orange-400 mb-3">Distillation Loss</h4>
            <p className="text-sm text-gray-300 mb-3">
              Combined loss function balancing knowledge distillation (KL divergence) 
              and standard classification (cross-entropy) objectives.
            </p>
            <div className="bg-orange-900/20 rounded p-2 text-xs font-mono text-orange-300">
              α * L_KD + (1-α) * L_CE
            </div>
          </div>
          
          <div className="bg-gray-900/50 rounded-lg p-4 border border-purple-800/30">
            <h4 className="font-semibold text-purple-400 mb-3">Feature Matching</h4>
            <p className="text-sm text-gray-300 mb-3">
              Advanced technique matching intermediate feature representations 
              between teacher and student networks for better knowledge transfer.
            </p>
            <div className="bg-purple-900/20 rounded p-2 text-xs font-mono text-purple-300">
              MSE(f_student, f_teacher)
            </div>
          </div>
          
          <div className="bg-gray-900/50 rounded-lg p-4 border border-red-800/30">
            <h4 className="font-semibold text-red-400 mb-3">Attention Transfer</h4>
            <p className="text-sm text-gray-300 mb-3">
              Transfer attention maps from teacher to student, helping the student 
              focus on the same important regions as the teacher.
            </p>
            <div className="bg-red-900/20 rounded p-2 text-xs font-mono text-red-300">
              ||A_s - A_t||_2
            </div>
          </div>
          
          <div className="bg-gray-900/50 rounded-lg p-4 border border-yellow-800/30">
            <h4 className="font-semibold text-yellow-400 mb-3">Progressive Distillation</h4>
            <p className="text-sm text-gray-300 mb-3">
              Multi-stage distillation process using intermediate-sized models 
              as stepping stones between teacher and final student.
            </p>
            <div className="bg-yellow-900/20 rounded p-2 text-xs font-mono text-yellow-300">
              Teacher → TA → Student
            </div>
          </div>
        </div>
      </motion.div>

      {/* Mathematical Formulation */}
      <motion.div 
        className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
        variants={itemVariants}
      >
        <h3 className="text-xl font-semibold mb-6">Mathematical Formulation</h3>
        
        <div className="space-y-6">
          <div className="bg-gray-900/50 rounded-lg p-6">
            <h4 className="font-semibold text-blue-400 mb-4">Complete Distillation Loss</h4>
            <div className="bg-gray-800 rounded-lg p-4 font-mono text-center mb-4">
              <div className="text-lg text-blue-400 mb-3">
                L<sub>total</sub> = α × L<sub>KD</sub>(σ(z<sub>s</sub>/T), σ(z<sub>t</sub>/T)) + (1-α) × L<sub>CE</sub>(σ(z<sub>s</sub>), y) + β × L<sub>feature</sub>
              </div>
            </div>
            
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div className="space-y-2">
                <p><strong className="text-blue-400">L<sub>KD</sub>:</strong> Knowledge Distillation Loss (KL Divergence)</p>
                <p><strong className="text-green-400">L<sub>CE</sub>:</strong> Cross-Entropy Loss with ground truth</p>
                <p><strong className="text-orange-400">L<sub>feature</sub>:</strong> Feature matching loss (optional)</p>
              </div>
              <div className="space-y-2">
                <p><strong className="text-purple-400">α:</strong> Distillation weight (0.7-0.9)</p>
                <p><strong className="text-red-400">T:</strong> Temperature parameter (3-5)</p>
                <p><strong className="text-yellow-400">β:</strong> Feature loss weight (0.1-0.3)</p>
              </div>
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-gray-900/50 rounded-lg p-4">
              <h4 className="font-semibold text-green-400 mb-3">Soft Target Generation</h4>
              <div className="bg-gray-800 rounded p-3 font-mono text-sm text-center">
                p<sub>i</sub> = exp(z<sub>i</sub>/T) / Σ<sub>j</sub> exp(z<sub>j</sub>/T)
              </div>
              <p className="text-xs text-gray-400 mt-2">
                Where z<sub>i</sub> are the logits and T is temperature
              </p>
            </div>
            
            <div className="bg-gray-900/50 rounded-lg p-4">
              <h4 className="font-semibold text-orange-400 mb-3">KL Divergence Loss</h4>
              <div className="bg-gray-800 rounded p-3 font-mono text-sm text-center">
                KL(P||Q) = Σ<sub>i</sub> P<sub>i</sub> log(P<sub>i</sub>/Q<sub>i</sub>)
              </div>
              <p className="text-xs text-gray-400 mt-2">
                Measures difference between teacher and student distributions
              </p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Applications & Use Cases */}
      <motion.div 
        className="bg-gray-800/50 rounded-lg p-6 border border-gray-700"
        variants={itemVariants}
      >
        <h3 className="text-xl font-semibold mb-6">Real-World Applications</h3>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-800/50">
            <h4 className="font-semibold text-blue-400 mb-2">Mobile Apps</h4>
            <p className="text-sm text-gray-300">Real-time image classification, object detection on smartphones</p>
          </div>
          
          <div className="bg-green-900/20 rounded-lg p-4 border border-green-800/50">
            <h4 className="font-semibold text-green-400 mb-2">IoT Devices</h4>
            <p className="text-sm text-gray-300">Edge computing, smart cameras, autonomous sensors</p>
          </div>
          
          <div className="bg-orange-900/20 rounded-lg p-4 border border-orange-800/50">
            <h4 className="font-semibold text-orange-400 mb-2">Autonomous Vehicles</h4>
            <p className="text-sm text-gray-300">Real-time perception, traffic sign recognition</p>
          </div>
          
          <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-800/50">
            <h4 className="font-semibold text-purple-400 mb-2">Medical Imaging</h4>
            <p className="text-sm text-gray-300">Portable diagnostic tools, point-of-care systems</p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default TheorySection;