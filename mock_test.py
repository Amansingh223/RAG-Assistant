from app.graph.workflow import app
import app.graph.nodes as nodes

class MockLLM:
    def invoke(self, prompt):
        class Response:
            def __init__(self, content): self.content = content
            
        if 'checking document relevance' in prompt: 
            return Response('no')
        if 'Rewrite the question' in prompt: 
            return Response('cricket sport')
        if 'Answer the question using the provided web context' in prompt: 
            return Response('Cricket is a bat-and-ball sport played between two teams of eleven players. [Answered via Web Search Fallback]')
            
        return Response('Generated answer.')

nodes.llm = MockLLM()

print("\nExecuting graph for 'What is Cricket?'...")
response = app.invoke({'question': 'What is Cricket?', 'rewritten': False})

print('\n\n--- FINAL RESPONSE ---\n')
print(response['generation'])
print('\n----------------------\n')
