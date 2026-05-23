#!/usr/bin/env python3
"""
SAFLA Complete AI Assistant - Full System Integration
====================================================

This example demonstrates a complete AI assistant built with SAFLA, showcasing
the full integration of all system components in a real-world application.

Learning Objectives:
- Integrate all SAFLA components in a cohesive application
- Implement conversation management with memory
- Add safety constraints and monitoring
- Track performance improvements over time
- Demonstrate enterprise-ready AI assistant patterns

Time to Complete: 45-60 minutes
Complexity: Advanced
"""

import asyncio
import time
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# SAFLA imports
from safla import (
    HybridMemoryArchitecture, 
    SAFLAConfig, 
    get_logger
)
from safla.core.safety_validation import OptimizedSafetyValidator
from safla.core.delta_evaluation import OptimizedDeltaEvaluator
from safla.core.meta_cognitive_engine import MetaCognitiveEngine

logger = get_logger(__name__)


class ConversationState(Enum):
    """States of a conversation."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class UserMessage:
    """User message in a conversation."""
    message_id: str
    user_id: str
    content: str
    timestamp: datetime
    intent: Optional[str] = None
    entities: Dict[str, Any] = None
    context: Dict[str, Any] = None


@dataclass
class AssistantResponse:
    """Assistant response to a user message."""
    response_id: str
    message_id: str
    content: str
    timestamp: datetime
    confidence: float
    reasoning: Optional[str] = None
    actions: List[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ConversationSession:
    """Complete conversation session."""
    session_id: str
    user_id: str
    started_at: datetime
    last_activity: datetime
    state: ConversationState
    messages: List[UserMessage]
    responses: List[AssistantResponse]
    context: Dict[str, Any]
    performance_metrics: Dict[str, float]


class AIAssistant:
    """Complete AI Assistant implementation using SAFLA."""
    
    def __init__(self, config: SAFLAConfig = None):
        self.config = config or SAFLAConfig()
        
        # Core SAFLA components
        self.memory = None
        self.safety_validator = None
        self.delta_evaluator = None
        self.meta_cognitive = None
        
        # Assistant-specific components
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.performance_history = []
        
        # Knowledge base and capabilities
        self.knowledge_base = {}
        self.capabilities = [
            "question_answering",
            "conversation",
            "task_assistance",
            "learning",
            "memory_management",
            "safety_monitoring"
        ]
    
    async def initialize(self):
        """Initialize all SAFLA components for the assistant."""
        print("🤖 Initializing SAFLA AI Assistant...")
        
        # Initialize core components
        self.memory = HybridMemoryArchitecture()
        await self.memory.start()
        print("  ✅ Memory system initialized")
        
        self.safety_validator = OptimizedSafetyValidator()
        print("  ✅ Safety validator initialized")
        
        self.delta_evaluator = OptimizedDeltaEvaluator()
        print("  ✅ Delta evaluator initialized")
        
        self.meta_cognitive = MetaCognitiveEngine()
        await self.meta_cognitive.start()
        print("  ✅ Meta-cognitive engine initialized")
        
        # Load knowledge base
        await self._load_knowledge_base()
        print("  ✅ Knowledge base loaded")
        
        print("🚀 AI Assistant ready for conversations!")
    
    async def start_conversation(self, user_id: str, initial_context: Dict[str, Any] = None) -> str:
        """Start a new conversation session."""
        session_id = str(uuid.uuid4())
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            started_at=datetime.now(),
            last_activity=datetime.now(),
            state=ConversationState.ACTIVE,
            messages=[],
            responses=[],
            context=initial_context or {},
            performance_metrics={}
        )
        
        self.active_sessions[session_id] = session
        
        # Store session start in episodic memory
        await self.memory.store_episodic_memory(
            content=f"Started conversation with user {user_id}",
            context={"user_id": user_id, "session_id": session_id},
            outcome="session_started",
            metadata={"timestamp": datetime.now().isoformat()}
        )
        
        # Load user profile if exists
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = await self._create_user_profile(user_id)
        
        print(f"💬 Started conversation session {session_id[:8]}... for user {user_id}")
        return session_id
    
    async def process_message(self, session_id: str, message_content: str) -> AssistantResponse:
        """Process a user message and generate a response."""
        start_time = time.time()
        
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Create user message
        message = UserMessage(
            message_id=str(uuid.uuid4()),
            user_id=session.user_id,
            content=message_content,
            timestamp=datetime.now()
        )
        
        # Safety validation
        safety_result = await self._validate_message_safety(message)
        if not safety_result.is_safe:
            return self._create_safety_response(message, safety_result)
        
        # Process with meta-cognitive awareness
        await self._update_metacognitive_state(session, message)
        
        # Analyze message intent and extract entities
        message.intent = await self._analyze_intent(message_content)
        message.entities = await self._extract_entities(message_content)
        
        # Retrieve relevant memories
        relevant_memories = await self._retrieve_relevant_memories(message)
        
        # Generate response
        response = await self._generate_response(message, session, relevant_memories)
        
        # Store conversation in memory
        await self._store_conversation_memory(message, response, session)
        
        # Update session
        session.messages.append(message)
        session.responses.append(response)
        session.last_activity = datetime.now()
        
        # Evaluate performance
        processing_time = time.time() - start_time
        await self._evaluate_response_performance(response, processing_time)
        
        # Learn from interaction
        await self._learn_from_interaction(message, response, session)
        
        print(f"💭 Processed message in {processing_time:.2f}s: '{message_content[:50]}...'")
        return response
    
    async def _validate_message_safety(self, message: UserMessage) -> Any:
        """Validate message for safety concerns."""
        request = {
            "operation": "message_processing",
            "data": {
                "content": message.content,
                "user_id": message.user_id,
                "content_length": len(message.content),
                "timestamp": message.timestamp.isoformat()
            },
            "metadata": {"message_id": message.message_id}
        }
        
        return await self.safety_validator.validate_request(request)
    
    def _create_safety_response(self, message: UserMessage, safety_result: Any) -> AssistantResponse:
        """Create a response for safety violations."""
        return AssistantResponse(
            response_id=str(uuid.uuid4()),
            message_id=message.message_id,
            content="I'm sorry, but I can't process that request due to safety concerns. Please rephrase your message.",
            timestamp=datetime.now(),
            confidence=1.0,
            reasoning="Safety validation failed",
            metadata={"safety_violation": True, "violations": len(safety_result.violations)}
        )
    
    async def _analyze_intent(self, content: str) -> str:
        """Analyze the intent of a message."""
        # Simplified intent detection (in production, use NLP models)
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["what", "how", "when", "where", "why", "?"]):
            return "question"
        elif any(word in content_lower for word in ["help", "assist", "support"]):
            return "help_request"
        elif any(word in content_lower for word in ["remember", "save", "store"]):
            return "memory_request"
        elif any(word in content_lower for word in ["hello", "hi", "hey"]):
            return "greeting"
        elif any(word in content_lower for word in ["bye", "goodbye", "exit"]):
            return "farewell"
        else:
            return "general"
    
    async def _extract_entities(self, content: str) -> Dict[str, Any]:
        """Extract entities from message content."""
        # Simplified entity extraction
        entities = {}
        
        # Date/time entities
        if "today" in content.lower():
            entities["date"] = datetime.now().date().isoformat()
        elif "tomorrow" in content.lower():
            entities["date"] = (datetime.now() + timedelta(days=1)).date().isoformat()
        
        # Number entities
        import re
        numbers = re.findall(r'\d+', content)
        if numbers:
            entities["numbers"] = [int(n) for n in numbers]
        
        return entities
    
    async def _retrieve_relevant_memories(self, message: UserMessage) -> List[Any]:
        """Retrieve memories relevant to the message."""
        # Generate embedding for the message (simplified)
        message_embedding = self._generate_text_embedding(message.content)
        
        # Search for similar memories
        similar_memories = await self.memory.search_similar_memories(
            query_embedding=message_embedding,
            top_k=5,
            similarity_threshold=0.7
        )
        
        # Also get recent episodic memories for this user
        recent_episodes = await self.memory.get_episodic_memories(
            start_time=datetime.now() - timedelta(hours=24),
            end_time=datetime.now(),
            limit=10
        )
        
        # Filter episodes for this user
        user_episodes = [ep for ep in recent_episodes 
                        if ep.context.get("user_id") == message.user_id]
        
        return {
            "similar_memories": similar_memories,
            "recent_episodes": user_episodes
        }
    
    async def _generate_response(self, message: UserMessage, session: ConversationSession, 
                               memories: Dict[str, Any]) -> AssistantResponse:
        """Generate a response based on message, context, and memories."""
        # Analyze conversation context
        conversation_context = self._analyze_conversation_context(session)
        
        # Generate response based on intent
        response_content = await self._generate_content_by_intent(
            message, conversation_context, memories
        )
        
        # Calculate confidence based on available information
        confidence = self._calculate_response_confidence(message, memories)
        
        # Determine actions to take
        actions = await self._determine_actions(message, response_content)
        
        response = AssistantResponse(
            response_id=str(uuid.uuid4()),
            message_id=message.message_id,
            content=response_content,
            timestamp=datetime.now(),
            confidence=confidence,
            reasoning=f"Generated based on intent: {message.intent}",
            actions=actions,
            metadata={
                "similar_memories_used": len(memories.get("similar_memories", [])),
                "recent_episodes_used": len(memories.get("recent_episodes", [])),
                "conversation_turn": len(session.messages) + 1
            }
        )
        
        return response
    
    async def _generate_content_by_intent(self, message: UserMessage, 
                                        context: Dict[str, Any], 
                                        memories: Dict[str, Any]) -> str:
        """Generate response content based on message intent."""
        intent = message.intent
        content = message.content
        
        if intent == "greeting":
            return f"Hello! I'm your SAFLA AI assistant. How can I help you today?"
        
        elif intent == "question":
            # Try to answer from knowledge base and memories
            knowledge_answer = await self._search_knowledge_base(content)
            if knowledge_answer:
                return f"Based on my knowledge: {knowledge_answer}"
            elif memories.get("similar_memories"):
                mem = memories["similar_memories"][0]
                return f"I recall something similar: {mem.content}"
            else:
                return "I'd be happy to help with that question. Could you provide more details?"
        
        elif intent == "help_request":
            capabilities_text = ", ".join(self.capabilities)
            return f"I can help you with: {capabilities_text}. What would you like assistance with?"
        
        elif intent == "memory_request":
            return "I'll remember that for you. My memory system will store this information for future reference."
        
        elif intent == "farewell":
            return "Goodbye! Feel free to return anytime if you need assistance."
        
        else:  # general
            if memories.get("recent_episodes"):
                return "I see we've been talking recently. How can I continue to help you?"
            else:
                return "I understand. Could you tell me more about what you'd like to achieve?"
    
    def _analyze_conversation_context(self, session: ConversationSession) -> Dict[str, Any]:
        """Analyze the context of the current conversation."""
        return {
            "session_duration": (datetime.now() - session.started_at).total_seconds(),
            "message_count": len(session.messages),
            "last_intent": session.messages[-1].intent if session.messages else None,
            "user_id": session.user_id,
            "session_state": session.state.value
        }
    
    def _calculate_response_confidence(self, message: UserMessage, memories: Dict[str, Any]) -> float:
        """Calculate confidence score for the response."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on available information
        if memories.get("similar_memories"):
            confidence += 0.2
        if memories.get("recent_episodes"):
            confidence += 0.1
        if message.intent in ["greeting", "farewell"]:
            confidence += 0.3  # High confidence for simple intents
        
        return min(confidence, 1.0)
    
    async def _determine_actions(self, message: UserMessage, response_content: str) -> List[str]:
        """Determine actions to take based on the message and response."""
        actions = []
        
        if message.intent == "memory_request":
            actions.append("store_user_information")
        elif message.intent == "question":
            actions.append("knowledge_search")
        elif "remember" in message.content.lower():
            actions.append("create_memory")
        
        return actions
    
    async def _store_conversation_memory(self, message: UserMessage, 
                                       response: AssistantResponse, 
                                       session: ConversationSession):
        """Store the conversation in various memory systems."""
        # Store in vector memory
        conversation_text = f"User: {message.content}\nAssistant: {response.content}"
        conversation_embedding = self._generate_text_embedding(conversation_text)
        
        await self.memory.store_vector_memory(
            content=conversation_text,
            embedding=conversation_embedding,
            metadata={
                "type": "conversation",
                "user_id": message.user_id,
                "session_id": session.session_id,
                "intent": message.intent,
                "confidence": response.confidence,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Store in episodic memory
        await self.memory.store_episodic_memory(
            content=f"Conversation turn: {message.content[:100]}...",
            context={
                "user_id": message.user_id,
                "session_id": session.session_id,
                "intent": message.intent,
                "response_confidence": response.confidence
            },
            outcome="conversation_turn_completed",
            metadata={
                "message_length": len(message.content),
                "response_length": len(response.content),
                "processing_time": (response.timestamp - message.timestamp).total_seconds()
            }
        )
    
    async def _evaluate_response_performance(self, response: AssistantResponse, processing_time: float):
        """Evaluate the performance of the response generation."""
        # Compare with previous performance
        baseline_time = 2.0  # 2 seconds baseline
        baseline_confidence = 0.7
        
        performance_delta = self.delta_evaluator.evaluate_delta(
            performance_data={
                "current_reward": response.confidence,
                "previous_reward": baseline_confidence,
                "tokens_used": len(response.content)
            },
            efficiency_data={
                "current_throughput": 1.0 / processing_time,
                "previous_throughput": 1.0 / baseline_time,
                "resource_used": processing_time
            },
            stability_data={
                "divergence_score": abs(processing_time - baseline_time) / baseline_time
            },
            context="conversation_response"
        )
        
        self.performance_history.append({
            "timestamp": datetime.now(),
            "response_id": response.response_id,
            "processing_time": processing_time,
            "confidence": response.confidence,
            "delta": performance_delta.overall_delta
        })
    
    async def _learn_from_interaction(self, message: UserMessage, 
                                    response: AssistantResponse, 
                                    session: ConversationSession):
        """Learn from the interaction to improve future responses."""
        # Update user profile
        user_profile = self.user_profiles[message.user_id]
        user_profile["interaction_count"] = user_profile.get("interaction_count", 0) + 1
        user_profile["last_interaction"] = datetime.now().isoformat()
        
        # Track intent patterns
        if "intent_history" not in user_profile:
            user_profile["intent_history"] = []
        user_profile["intent_history"].append(message.intent)
        
        # Update meta-cognitive understanding
        await self.meta_cognitive.update_knowledge(
            experience=f"Processed {message.intent} intent with {response.confidence:.2f} confidence",
            outcome="successful_interaction" if response.confidence > 0.7 else "low_confidence_interaction",
            context={"user_id": message.user_id, "session_id": session.session_id}
        )
    
    async def _update_metacognitive_state(self, session: ConversationSession, message: UserMessage):
        """Update meta-cognitive state based on the conversation."""
        # Analyze conversation patterns
        conversation_analysis = {
            "user_engagement": len(session.messages) > 3,
            "session_duration": (datetime.now() - session.started_at).total_seconds(),
            "message_complexity": len(message.content),
            "intent_variety": len(set(m.intent for m in session.messages if m.intent))
        }
        
        # Update meta-cognitive awareness
        await self.meta_cognitive.assess_current_state(conversation_analysis)
    
    async def _search_knowledge_base(self, query: str) -> Optional[str]:
        """Search the knowledge base for relevant information."""
        # Simplified knowledge base search
        query_lower = query.lower()
        
        knowledge_items = {
            "safla": "SAFLA is a Self-Aware Feedback Loop Algorithm that provides sophisticated AI/ML capabilities with hybrid memory architecture.",
            "memory": "SAFLA uses a hybrid memory architecture with vector, episodic, semantic, and working memory components.",
            "safety": "SAFLA includes comprehensive safety validation with constraints, risk assessment, and monitoring.",
            "performance": "SAFLA tracks performance improvements using delta evaluation across multiple dimensions."
        }
        
        for key, value in knowledge_items.items():
            if key in query_lower:
                return value
        
        return None
    
    def _generate_text_embedding(self, text: str) -> List[float]:
        """Generate a simple text embedding (in production, use proper models)."""
        # Simplified embedding generation
        import hashlib
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Convert hash to 512-dimensional vector
        embedding = []
        for i in range(512):
            char_index = i % len(text_hash)
            embedding.append((ord(text_hash[char_index]) - 48) / 15.0 - 1.0)  # Normalize to [-1, 1]
        
        return embedding
    
    async def _create_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Create a new user profile."""
        profile = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "interaction_count": 0,
            "preferences": {},
            "capabilities_used": [],
            "intent_history": []
        }
        
        # Store profile in semantic memory
        await self.memory.add_semantic_node(
            content=f"User profile for {user_id}",
            node_type="user_profile",
            properties=profile
        )
        
        return profile
    
    async def _load_knowledge_base(self):
        """Load the assistant's knowledge base."""
        # Load core SAFLA knowledge
        safla_knowledge = [
            ("SAFLA Architecture", {
                "description": "Hybrid memory system with multiple specialized components",
                "components": ["vector_memory", "episodic_memory", "semantic_memory", "working_memory"]
            }),
            ("Safety Features", {
                "description": "Comprehensive safety validation and monitoring",
                "features": ["constraint_checking", "risk_assessment", "emergency_procedures"]
            }),
            ("Performance Tracking", {
                "description": "Delta evaluation system for quantifying improvements",
                "metrics": ["performance", "efficiency", "stability", "capability"]
            })
        ]
        
        for topic, details in safla_knowledge:
            await self.memory.add_semantic_node(
                content=topic,
                node_type="knowledge",
                properties=details
            )
            self.knowledge_base[topic.lower().replace(" ", "_")] = details
    
    async def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the conversation session."""
        session = self.active_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        # Calculate conversation metrics
        total_messages = len(session.messages)
        avg_confidence = sum(r.confidence for r in session.responses) / max(len(session.responses), 1)
        intents = [m.intent for m in session.messages if m.intent]
        unique_intents = len(set(intents))
        
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "duration": (session.last_activity - session.started_at).total_seconds(),
            "total_messages": total_messages,
            "average_confidence": avg_confidence,
            "unique_intents": unique_intents,
            "state": session.state.value,
            "last_activity": session.last_activity.isoformat()
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall assistant performance metrics."""
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        avg_processing_time = sum(p["processing_time"] for p in self.performance_history) / len(self.performance_history)
        avg_confidence = sum(p["confidence"] for p in self.performance_history) / len(self.performance_history)
        avg_delta = sum(p["delta"] for p in self.performance_history) / len(self.performance_history)
        
        return {
            "total_interactions": len(self.performance_history),
            "average_processing_time": avg_processing_time,
            "average_confidence": avg_confidence,
            "average_improvement_delta": avg_delta,
            "active_sessions": len(self.active_sessions),
            "total_users": len(self.user_profiles)
        }
    
    async def cleanup(self):
        """Clean up all resources."""
        print("\n🧹 Cleaning up AI Assistant...")
        
        if self.memory:
            await self.memory.stop()
        if self.safety_validator:
            await self.safety_validator.close()
        if self.meta_cognitive:
            await self.meta_cognitive.stop()
        
        print("✅ AI Assistant cleanup complete!")


async def demonstrate_ai_assistant():
    """Demonstrate the complete AI assistant."""
    print("🤖 SAFLA Complete AI Assistant Demonstration")
    print("=" * 65)
    
    # Initialize assistant
    assistant = AIAssistant()
    await assistant.initialize()
    
    try:
        # Start a conversation
        session_id = await assistant.start_conversation("demo_user_123")
        
        # Simulate a complete conversation
        conversation_flow = [
            "Hello! I'm new here, can you help me?",
            "What is SAFLA and what can it do?",
            "How does the memory system work?",
            "Can you remember that I'm interested in AI safety?",
            "What are the safety features in SAFLA?",
            "Show me some performance metrics",
            "Thank you, that was very helpful!"
        ]
        
        print(f"\n💬 Starting conversation simulation...")
        
        for i, user_message in enumerate(conversation_flow, 1):
            print(f"\n--- Turn {i} ---")
            print(f"👤 User: {user_message}")
            
            response = await assistant.process_message(session_id, user_message)
            
            print(f"🤖 Assistant: {response.content}")
            print(f"   📊 Confidence: {response.confidence:.2f}")
            print(f"   🎯 Intent: {assistant.active_sessions[session_id].messages[-1].intent}")
            
            if response.actions:
                print(f"   ⚡ Actions: {', '.join(response.actions)}")
            
            # Small delay to simulate natural conversation
            await asyncio.sleep(0.5)
        
        # Show conversation summary
        print("\n📋 Conversation Summary")
        print("-" * 30)
        summary = await assistant.get_conversation_summary(session_id)
        for key, value in summary.items():
            if key != "session_id":
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Show performance metrics
        print("\n📊 Assistant Performance Metrics")
        print("-" * 40)
        metrics = await assistant.get_performance_metrics()
        for key, value in metrics.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("\n🎉 AI Assistant demonstration completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Complete conversation management with memory")
        print("  • Intent recognition and entity extraction")
        print("  • Safety validation for all interactions")
        print("  • Performance tracking and improvement measurement")
        print("  • Meta-cognitive awareness and learning")
        print("  • Knowledge base integration")
        print("  • User profile management")
        
    except Exception as e:
        logger.exception("AI Assistant demo failed")
        print(f"❌ Demo failed: {e}")
    
    finally:
        await assistant.cleanup()


if __name__ == "__main__":
    asyncio.run(demonstrate_ai_assistant())