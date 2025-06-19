import os
import sys
import logging
from datetime import datetime
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"model_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_attention_visualization():
    """Run attention visualization analysis"""
    logger.info("Running attention visualization analysis...")
    try:
        from results.attention_visualization import main as attention_main
        attention_main()
        logger.info("Attention visualization completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error in attention visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_error_analysis():
    logger.info("Running error analysis...")
    try:
        from results.error_analysis import main as error_main
        error_main()
        logger.info("Error analysis completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error in error analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_shap_analysis():
    """Run SHAP analysis"""
    logger.info("Running SHAP analysis...")
    try:
        from results.shap_analysis import main as shap_main
        shap_main()
        logger.info("SHAP analysis completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error in SHAP analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def install_dependencies():
    """Install required dependencies for the analyses"""
    logger.info("Installing required dependencies...")
    
    try:
        import subprocess
        
        subprocess.check_call([sys.executable, "-m", "pip", "install", "bertviz"])
        logger.info("Installed bertviz")
        
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
        logger.info("Installed shap")
        
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "seaborn"])
        logger.info("Installed pandas and seaborn")
     
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        logger.info("Installed tabulate for Markdown tables")
        
        logger.info("All dependencies installed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run all analyses"""
    parser = argparse.ArgumentParser(description="Run model interpretability and analysis tools")
    parser.add_argument("--skip-dependencies", action="store_true", help="Skip installing dependencies")
    parser.add_argument("--attention-only", action="store_true", help="Run only attention visualization")
    parser.add_argument("--error-only", action="store_true", help="Run only error analysis")
    parser.add_argument("--shap-only", action="store_true", help="Run only SHAP analysis")
    parser.add_argument("--run-all", action="store_true", help="Run all analyses including cross-lingual")
    
    args = parser.parse_args()
    
    if not args.skip_dependencies:
        if not install_dependencies():
            logger.error("Failed to install dependencies. Exiting.")
            return 1

    run_base = not (args.attention_only or args.error_only or args.shap_only)
    run_all = args.run_all
    
    
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "error_analysis"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shap_analysis"), exist_ok=True)
    
    success = True

    if run_base or run_all or args.attention_only:
        success = success and run_attention_visualization()
    
    if run_base or run_all or args.error_only:
        success = success and run_error_analysis()
    
    if run_base or run_all or args.shap_only:
        success = success and run_shap_analysis()
    
    
    if success:
        logger.info("All requested analyses completed successfully!")
        return 0
    else:
        logger.error("One or more analyses failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
