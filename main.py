"""
CoVulPecker - Vulnerability Detection Pipeline

Sử dụng LangGraph để phát hiện lỗ hổng bảo mật trong code C/C++.
"""
import sys
from pathlib import Path

from src.graph import vulnerability_detector
from src.config import config
from src.report import generate_report, print_report
from src.logger import logger


def analyze_code(source_code: str, max_iterations: int = 3, source_file: str = None) -> dict:
    """
    Phân tích source code để phát hiện lỗ hổng.
    
    Args:
        source_code: Source code C/C++ cần phân tích
        max_iterations: Số lần tối đa Critic có thể reject
        source_file: Đường dẫn file gốc (để logging)
        
    Returns:
        Final state sau khi chạy pipeline
    """
    # Start logging
    logger.start_run(source_code=source_code, source_file=source_file)
    
    initial_state = {
        "source_code": source_code,
        "classification": None,
        "detection": None,
        "reasoning": None,
        "critic": None,
        "critic_feedback_history": [],
        "verification": None,
        "iteration_count": 1,
        "max_iterations": max_iterations,
        "current_stage": "start",
        "is_complete": False,
    }
    
    print("🚀 Starting vulnerability detection pipeline...")
    print(f"📡 Using LLM API: {config.LLM_API_BASE_URL}")
    print("-" * 50)
    
    # Run the graph
    final_state = None
    for step in vulnerability_detector.stream(initial_state):
        # Get the node name and state update
        for node_name, state_update in step.items():
            current_stage = state_update.get("current_stage", node_name)
            print(f"✓ Completed: {current_stage.upper()}")
            final_state = state_update
    
    return final_state


def analyze_file(file_path: str, max_iterations: int = 3) -> dict:
    """
    Phân tích file source code.
    
    Args:
        file_path: Đường dẫn tới file C/C++
        max_iterations: Số lần tối đa Critic có thể reject
        
    Returns:
        Final state sau khi chạy pipeline
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    source_code = path.read_text()
    print(f"📄 Analyzing file: {file_path}")
    print(f"📏 File size: {len(source_code)} characters")
    
    return analyze_code(source_code, max_iterations, source_file=file_path)


def main():
    """Main entry point."""
    # Default to example.c if no argument provided
    file_path = sys.argv[1] if len(sys.argv) > 1 else "example.c"
    
    try:
        # Start fresh logger for this run
        source_code = Path(file_path).read_text()
        logger.start_run(source_code=source_code, source_file=file_path)
        
        print(f"📄 Analyzing file: {file_path}")
        print(f"📏 File size: {len(source_code)} characters")
        print("🚀 Starting vulnerability detection pipeline...")
        print(f"📡 Using LLM API: {config.LLM_API_BASE_URL}")
        print("-" * 50)
        
        initial_state = {
            "source_code": source_code,
            "classification": None,
            "detection": None,
            "reasoning": None,
            "critic": None,
            "critic_feedback_history": [],
            "verification": None,
            "iteration_count": 1,
            "max_iterations": config.MAX_CRITIC_ITERATIONS,
            "current_stage": "start",
            "is_complete": False,
        }
        
        # Run and collect final state with streaming
        result = None
        for step in vulnerability_detector.stream(initial_state):
            for node_name, state_update in step.items():
                current_stage = state_update.get("current_stage", node_name)
                print(f"✓ Completed: {current_stage.upper()}")
                result = state_update
        
        if result and result.get("is_complete"):
            # Get full result with invoke for report
            full_result = vulnerability_detector.invoke(initial_state)
            
            # Generate report
            report = generate_report(full_result)
            print("\n")
            print(print_report(report))
            
            # End logging and save
            logger.end_run(final_result={
                "is_complete": True,
                "classification": str(full_result.get("classification")),
                "vulnerabilities_count": len(full_result.get("detection").vulnerabilities) if full_result.get("detection") else 0,
                "critic_decision": str(full_result.get("critic").decision) if full_result.get("critic") else None,
                "vulnerability_confirmed": full_result.get("verification").vulnerability_confirmed if full_result.get("verification") else None
            })
            
            # Save log files
            saved_files = logger.save_log(output_dir="logs", format="both")
            print("\n" + "=" * 50)
            print("📝 CONVERSATION LOG SAVED:")
            for f in saved_files:
                print(f"   📄 {f}")
            print("=" * 50)
        else:
            logger.end_run(final_result={"is_complete": False, "error": "Pipeline did not complete"})
            logger.save_log(output_dir="logs", format="both")
            print("\n⚠️ Pipeline did not complete successfully")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.end_run(final_result={"is_complete": False, "error": str(e)})
        logger.save_log(output_dir="logs", format="both")
        print(f"❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()

