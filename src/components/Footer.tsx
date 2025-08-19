import React from 'react';
import { Heart, Github, Linkedin, Mail } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-gray-900 border-t border-gray-800 mt-16">
      <div className="container mx-auto px-6 py-8">
        <div className="flex flex-col md:flex-row items-center justify-between">
          <div className="flex items-center space-x-2 mb-4 md:mb-0">
            <span className="text-gray-400">Made with</span>
            <Heart className="w-4 h-4 text-red-500 fill-current" />
            <span className="text-gray-400">for ML Education</span>
          </div>
          
          <div className="flex items-center space-x-6">
            <a 
              href="#" 
              className="text-gray-400 hover:text-white transition-colors"
              aria-label="GitHub"
            >
              <Github className="w-5 h-5" />
            </a>
            <a 
              href="#" 
              className="text-gray-400 hover:text-white transition-colors"
              aria-label="LinkedIn"
            >
              <Linkedin className="w-5 h-5" />
            </a>
            <a 
              href="#" 
              className="text-gray-400 hover:text-white transition-colors"
              aria-label="Email"
            >
              <Mail className="w-5 h-5" />
            </a>
          </div>
        </div>
        
        <div className="mt-6 pt-6 border-t border-gray-800 text-center text-gray-400 text-sm">
          <p>© 2025 Knowledge Distillation Platform. Educational content for machine learning research and development.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;