from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate

class PromptTemplates:
    """Class to manage prompt templates for the RAG chatbot"""
    
    def __init__(self):
        """Initialize prompt templates"""
        self.templates = self._create_templates()
    
    def _create_templates(self) -> dict:
        """Create all prompt templates"""
        templates = {}
        
        # Basic RAG prompt template
        rag_template = """You are a helpful AI assistant that answers questions based on the provided context. 
        Use only the information from the context to answer the question. If the context doesn't contain enough 
        information to answer the question, say "I don't have enough information to answer this question based on the provided context."

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        templates['rag_basic'] = PromptTemplate(
            input_variables=["context", "question"],
            template=rag_template
        )
        
       
        chat_rag_template = """You are a helpful AI assistant that answers questions based on the provided context. 
        Use only the information from the context to answer the question. If the context doesn't contain enough 
        information to answer the question, say "I don't have enough information to answer this question based on the provided context."

        Context:
        {context}

        Human: {question}
        Assistant:"""
        
        templates['rag_chat'] = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a helpful AI assistant that answers questions based on the provided context. "
                "Use only the information from the context to answer the question. If the context doesn't contain enough "
                "information to answer the question, say 'I don't have enough information to answer this question based on the provided context.'"
            ),
            HumanMessagePromptTemplate.from_template(
                "Context:\n{context}\n\nQuestion: {question}"
            )
        ])
        
       
        summary_template = """You are a helpful AI assistant that summarizes documents. 
        Please provide a concise summary of the following text, highlighting the key points and main ideas.

        Text to summarize:
        {text}

        Summary:"""
        
        templates['summary'] = PromptTemplate(
            input_variables=["text"],
            template=summary_template
        )
        
        
        question_gen_template = """Based on the following context, generate 3 relevant questions that could be asked about this information.
        Make sure the questions are specific and would require the context to answer properly.

        Context:
        {context}

        Generated Questions:
        1. """
        
        templates['question_generation'] = PromptTemplate(
            input_variables=["context"],
            template=question_gen_template
        )
        
        fact_check_template = """You are a helpful AI assistant that fact-checks information. 
        Based on the provided context, determine if the following statement is true, false, or if there's insufficient information.

        Context:
        {context}

        Statement to fact-check: {statement}

        Analysis:"""
        
        templates['fact_check'] = PromptTemplate(
            input_variables=["context", "statement"],
            template=fact_check_template
        )
       
        detailed_analysis_template = """You are a helpful AI assistant that provides detailed analysis. 
        Based on the provided context, please provide a comprehensive analysis of the following question.
        Include relevant details, examples, and connections from the context.

        Context:
        {context}

        Question: {question}

        Detailed Analysis:"""
        
        templates['detailed_analysis'] = PromptTemplate(
            input_variables=["context", "question"],
            template=detailed_analysis_template
        )
        
        return templates
    
    def get_template(self, template_name: str) -> PromptTemplate:
        """Get a specific prompt template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Available templates: {list(self.templates.keys())}")
        return self.templates[template_name]
    
    def get_available_templates(self) -> list:
        """Get list of available template names"""
        return list(self.templates.keys())
    
    def format_rag_prompt(self, context: str, question: str, template_name: str = 'rag_basic') -> str:
        """Format a RAG prompt with context and question"""
        template = self.get_template(template_name)
        return template.format(context=context, question=question)
    
    def format_summary_prompt(self, text: str) -> str:
        """Format a summarization prompt"""
        template = self.get_template('summary')
        return template.format(text=text)
    
    def format_question_gen_prompt(self, context: str) -> str:
        """Format a question generation prompt"""
        template = self.get_template('question_generation')
        return template.format(context=context)
    
    def format_fact_check_prompt(self, context: str, statement: str) -> str:
        """Format a fact-checking prompt"""
        template = self.get_template('fact_check')
        return template.format(context=context, statement=statement)
    
    def format_detailed_analysis_prompt(self, context: str, question: str) -> str:
        """Format a detailed analysis prompt"""
        template = self.get_template('detailed_analysis')
        return template.format(context=context, question=question)
    
    def print_template_info(self):
        """Print information about available templates"""
        print("\nAvailable Prompt Templates:")
        for template_name in self.templates:
            template = self.templates[template_name]
            input_vars = template.input_variables
            print(f"   {template_name}: {input_vars}")
    
    def create_custom_template(self, template_name: str, template_text: str, input_variables: list) -> PromptTemplate:
        """Create a custom prompt template"""
        try:
            custom_template = PromptTemplate(
                input_variables=input_variables,
                template=template_text
            )
            self.templates[template_name] = custom_template
            print(f"Custom template '{template_name}' created successfully")
            return custom_template
        except Exception as e:
            print(f"Error creating custom template: {e}")
            raise
