"""Logging module for capturing agent conversations."""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ConversationEntry:
    """A single conversation entry between user and agent."""
    agent_name: str
    timestamp: str
    system_prompt: str
    user_message: str
    llm_response: str
    parsed_result: Optional[dict] = None
    error: Optional[str] = None


@dataclass 
class PipelineLog:
    """Complete log of a pipeline run."""
    run_id: str
    start_time: str
    end_time: Optional[str] = None
    source_file: Optional[str] = None
    source_code: str = ""
    conversations: list = field(default_factory=list)
    final_result: Optional[dict] = None
    
    def add_conversation(self, entry: ConversationEntry):
        """Add a conversation entry to the log."""
        self.conversations.append(asdict(entry))
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_markdown(self) -> str:
        """Convert to readable markdown format."""
        lines = []
        lines.append(f"# Pipeline Run Log: {self.run_id}")
        lines.append(f"")
        lines.append(f"**Start Time:** {self.start_time}")
        lines.append(f"**End Time:** {self.end_time or 'N/A'}")
        lines.append(f"**Source File:** {self.source_file or 'N/A'}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Source code
        lines.append("## Source Code")
        lines.append("")
        lines.append("```c")
        lines.append(self.source_code)
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Conversations
        lines.append("## Agent Conversations")
        lines.append("")
        
        for i, conv in enumerate(self.conversations, 1):
            lines.append(f"### {i}. {conv['agent_name'].upper()} Agent")
            lines.append(f"")
            lines.append(f"**Timestamp:** {conv['timestamp']}")
            lines.append("")
            
            # System prompt
            lines.append("#### System Prompt")
            lines.append("```")
            lines.append(conv['system_prompt'])
            lines.append("```")
            lines.append("")
            
            # User message
            lines.append("#### User Message (Input)")
            lines.append("```")
            lines.append(conv['user_message'])
            lines.append("```")
            lines.append("")
            
            # LLM Response
            lines.append("#### LLM Response")
            lines.append("```")
            lines.append(conv['llm_response'])
            lines.append("```")
            lines.append("")
            
            # Parsed result
            if conv.get('parsed_result'):
                lines.append("#### Parsed Result")
                lines.append("```json")
                lines.append(json.dumps(conv['parsed_result'], indent=2, ensure_ascii=False))
                lines.append("```")
                lines.append("")
            
            # Error
            if conv.get('error'):
                lines.append(f"#### ⚠️ Error")
                lines.append(f"```")
                lines.append(conv['error'])
                lines.append("```")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        # Final result
        if self.final_result:
            lines.append("## Final Result")
            lines.append("```json")
            lines.append(json.dumps(self.final_result, indent=2, ensure_ascii=False, default=str))
            lines.append("```")
        
        return "\n".join(lines)


class ConversationLogger:
    """Singleton logger for capturing agent conversations."""
    
    _instance: Optional['ConversationLogger'] = None
    _current_log: Optional[PipelineLog] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'ConversationLogger':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton instance."""
        cls._instance = None
        cls._current_log = None
    
    def start_run(self, source_code: str, source_file: Optional[str] = None) -> str:
        """Start a new pipeline run and return the run ID."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_log = PipelineLog(
            run_id=run_id,
            start_time=datetime.now().isoformat(),
            source_file=source_file,
            source_code=source_code
        )
        return run_id
    
    def log_conversation(
        self,
        agent_name: str,
        system_prompt: str,
        user_message: str,
        llm_response: str,
        parsed_result: Optional[dict] = None,
        error: Optional[str] = None
    ):
        """Log a conversation between user and agent."""
        if self._current_log is None:
            return
        
        entry = ConversationEntry(
            agent_name=agent_name,
            timestamp=datetime.now().isoformat(),
            system_prompt=system_prompt,
            user_message=user_message,
            llm_response=llm_response,
            parsed_result=parsed_result,
            error=error
        )
        self._current_log.add_conversation(entry)
    
    def end_run(self, final_result: Optional[dict] = None):
        """End the current pipeline run."""
        if self._current_log:
            self._current_log.end_time = datetime.now().isoformat()
            self._current_log.final_result = final_result
    
    def save_log(self, output_dir: str = "logs", format: str = "both") -> list[str]:
        """
        Save the log to file(s).
        
        Args:
            output_dir: Directory to save logs
            format: 'json', 'markdown', or 'both'
            
        Returns:
            List of saved file paths
        """
        if self._current_log is None:
            return []
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        run_id = self._current_log.run_id
        
        if format in ('json', 'both'):
            json_file = output_path / f"conversation_{run_id}.json"
            json_file.write_text(self._current_log.to_json(), encoding='utf-8')
            saved_files.append(str(json_file))
        
        if format in ('markdown', 'both'):
            md_file = output_path / f"conversation_{run_id}.md"
            md_file.write_text(self._current_log.to_markdown(), encoding='utf-8')
            saved_files.append(str(md_file))
        
        return saved_files
    
    def get_current_log(self) -> Optional[PipelineLog]:
        """Get the current log."""
        return self._current_log


# Global logger instance
logger = ConversationLogger.get_instance()


