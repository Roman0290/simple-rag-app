from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from utils.config_loader import get_groq_api_key, get_groq_model_name, get_temperature
from .prompt_templates import PromptTemplates

class LLMChain:
    """Class to handle LLM generation using Groq"""
    
    def __init__(self, model_name: str = None, temperature: float = None):
        """Initialize LLM chain with Groq"""
        self.model_name = model_name or get_groq_model_name()
        self.temperature = temperature or get_temperature()
        self.llm = None
        self.prompt_templates = PromptTemplates()
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the Groq LLM"""
        try:
            groq_api_key = get_groq_api_key()
            self.llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=self.model_name,
                temperature=self.temperature
            )
            print(f"Groq LLM initialized successfully")
            print(f"   Model: {self.model_name}")
            print(f"   Temperature: {self.temperature}")
        except Exception as e:
            print(f"❌ Error initializing Groq LLM: {e}")
            raise
    
    def generate_answer(self, query: str, context: str, template_name: str = 'rag_basic') -> str:
        """Generate answer using the LLM with context"""
        try:
            # Format the prompt using the specified template
            if template_name == 'rag_chat':
                prompt = self.prompt_templates.get_template(template_name)
                formatted_prompt = prompt.format_messages(context=context, question=query)
                response = self.llm.invoke(formatted_prompt)
                return response.content
            else:
                prompt = self.prompt_templates.get_template(template_name)
                formatted_prompt = prompt.format(context=context, question=query)
                response = self.llm.invoke(formatted_prompt)
                return response.content
                
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return f"Sorry, I encountered an error while generating the answer: {str(e)}"
    
    def generate_summary(self, text: str) -> str:
        """Generate a summary of the given text"""
        try:
            prompt = self.prompt_templates.get_template('summary')
            formatted_prompt = prompt.format(text=text)
            response = self.llm.invoke(formatted_prompt)
            return response.content
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return f"Sorry, I encountered an error while generating the summary: {str(e)}"
    
    def generate_questions(self, context: str) -> str:
        """Generate questions based on the given context"""
        try:
            prompt = self.prompt_templates.get_template('question_generation')
            formatted_prompt = prompt.format(context=context)
            response = self.llm.invoke(formatted_prompt)
            return response.content
        except Exception as e:
            print(f"❌ Error generating questions: {e}")
            return f"Sorry, I encountered an error while generating questions: {str(e)}"
    
    def fact_check(self, context: str, statement: str) -> str:
        """Fact-check a statement against the given context"""
        try:
            prompt = self.prompt_templates.get_template('fact_check')
            formatted_prompt = prompt.format(context=context, statement=statement)
            response = self.llm.invoke(formatted_prompt)
            return response.content
        except Exception as e:
            print(f"❌ Error during fact-checking: {e}")
            return f"Sorry, I encountered an error while fact-checking: {str(e)}"
    
    def detailed_analysis(self, context: str, question: str) -> str:
        """Provide detailed analysis based on the given context"""
        try:
            prompt = self.prompt_templates.get_template('detailed_analysis')
            formatted_prompt = prompt.format(context=context, question=question)
            response = self.llm.invoke(formatted_prompt)
            return response.content
        except Exception as e:
            print(f"❌ Error during detailed analysis: {e}")
            return f"Sorry, I encountered an error while providing analysis: {str(e)}"
    
    def create_retrieval_qa_chain(self, retriever) -> RetrievalQA:
        """Create a RetrievalQA chain"""
        try:
            chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": self.prompt_templates.get_template('rag_basic')
                }
            )
            print("RetrievalQA chain created successfully")
            return chain
        except Exception as e:
            print(f"❌ Error creating RetrievalQA chain: {e}")
            raise
    
    def create_custom_chain(self, retriever) -> Any:
        """Create a custom chain using LCEL (LangChain Expression Language)"""
        try:
            # Create a custom chain that formats context and generates answers
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | self.prompt_templates.get_template('rag_basic')
                | self.llm
                | StrOutputParser()
            )
            
            print("Custom LCEL chain created successfully")
            return rag_chain
        except Exception as e:
            print(f"❌ Error creating custom chain: {e}")
            raise
    
    def process_query_with_chain(self, query: str, chain, include_sources: bool = False) -> Dict[str, Any]:
        """Process a query using a chain and return results"""
        try:
            if hasattr(chain, 'invoke'):
                # For custom chains
                result = chain.invoke(query)
                return {
                    "answer": result,
                    "sources": [],
                    "query": query
                }
            elif hasattr(chain, '__call__'):
                # For RetrievalQA chains
                result = chain(query)
                return {
                    "answer": result.get("result", ""),
                    "sources": result.get("source_documents", []),
                    "query": query
                }
            else:
                raise ValueError("Invalid chain type")
                
        except Exception as e:
            print(f"❌ Error processing query with chain: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": [],
                "query": query
            }
    
    def update_llm_parameters(self, model_name: str = None, temperature: float = None):
        """Update LLM parameters"""
        if model_name is not None:
            self.model_name = model_name
        if temperature is not None:
            self.temperature = temperature
        
        # Reinitialize LLM with new parameters
        self._initialize_llm()
        print(f"Updated LLM parameters: model={self.model_name}, temperature={self.temperature}")
    
    def get_llm_info(self) -> Dict[str, Any]:
        """Get information about the current LLM configuration"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "provider": "Groq",
            "available_templates": self.prompt_templates.get_available_templates()
        }
    
    def print_llm_summary(self):
        """Print a summary of the LLM configuration"""
        info = self.get_llm_info()
        print("\n🤖 LLM Configuration Summary:")
        print(f"   Provider: {info['provider']}")
        print(f"   Model: {info['model_name']}")
        print(f"   Temperature: {info['temperature']}")
        print(f"   Available templates: {', '.join(info['available_templates'])}")
    
    def test_llm_connection(self) -> bool:
        """Test if the LLM is working properly"""
        try:
            test_prompt = "Hello! Please respond with 'Connection test successful.'"
            response = self.llm.invoke(test_prompt)
            if "Connection test successful" in response.content:
                print("LLM connection test successful")
                return True
            else:
                print("⚠️ LLM connection test returned unexpected response")
                return False
        except Exception as e:
            print(f"❌ LLM connection test failed: {e}")
            return False
