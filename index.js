import {tool} from '@langchain/core/tools'
import { z } from 'zod/v4'
import {ChatCohere} from '@langchain/cohere'
import 'dotenv/config'

import { MessagesAnnotation ,StateGraph} from '@langchain/langgraph'

const llm = new ChatCohere({
    model:'command-r-plus',
    temperature:0,
    apiKey:process.env.COHERE_KEY
})
console.log(llm)
const multiply = tool(async ({ a, b }) => a * b, {
  name: "multiply",
  description: "multiply two numbers",
  schema: z.object({
    a: z.number().describe("the first number"),
    b: z.number().describe("the second number"),
  }),
})

const divide = tool(async({a,b})=> a/b,{
    name:"division",
    description:'divide two numbers',
    schema: z.object({ 
        a:z.number().describe('the first number'),
        b:z.number().describe('the second number')
    })
})


const add = tool(async({a,b})=>a+b,{
    name:"add",
    description:'Add two numbers',
    schema: z.object({
        a:z.number().describe('the first number'),
        b:z.number().describe('the second number')
    })
})

const tools = [multiply,divide,add]
const toolsByName = Object.fromEntries(tools.map((t)=>[t.name,t]))



const bindTools = llm.bindTools(tools)

export const llmCalls = async(state)=>{
    const result = await bindTools.invoke([
        {
            role:'user',
            content:'You are a helpfull assistant taked with priority to perform arithmetic operations on two numbers'
        },
        ...state.messages,
    ])
    return{
        messages: [result]
    }
}


async function toolNode(state) {
    const results = []
    const lastMessage = state.messages.at(-1)
    if(lastMessage?.tool_calls?.length){
        for(const toolCall of lastMessage.tool_calls){
            const tools = toolsByName[toolCall.name]
            const observation = await tools.invoke(toolCall.args)
            results.push(
                {
                    type:'tool',
                    content:observation,
                    tool_call_id:toolCall.id
                })
            
    
        }
    } 
    return {messages:results}
    
}

function shouldContinue(state){
    const messages = state.messages
    const lastMessage = messages.at(-1)
     if(lastMessage?.tool_calls?.length){
        return "Action"
     }
     return '__end__'
}


const agentBuilder = new StateGraph(MessagesAnnotation)
.addNode("llmCall",llmCalls)
.addNode("tool",toolNode)
.addEdge("__start__","llmCall")
.addConditionalEdges("llmCall",shouldContinue,{
    "Action":"tool",
    "__end__":"__end__"

})
.addEdge("tool","llmCall")
.compile()

const message = [
    {
        role:'user',
        content:"What is addition of 12 and 34"
    }
]  

const result = await agentBuilder.invoke({messages:message})
console.log(result.messages)