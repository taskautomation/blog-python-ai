

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


from langchain.agents.agent_toolkits import FileManagementToolkit
import os

# Load tile management toolkit
toolkit = FileManagementToolkit(
    selected_tools=["write_file", "list_directory", "read_file"],
) 

# Set working directory to your Jekyll blog directory
os.chdir('C:\\Users\\fredr\jekyll\\jekyll-ai-blog\\')


from langchain.tools import tool
import subprocess

# Define git helper functions
@tool
def git_add(file_path="."):
    """Adds file(s) to the staging area."""
    subprocess.check_call(["git", "add", file_path])

@tool
def git_commit(message):
    """Commits the changes with a given message."""
    subprocess.check_call(["git", "commit", "-m", message])

@tool
def git_push(branch_name="gh-pages"):
    """Pushes the committed changes to a specified remote and branch."""
    subprocess.check_call(["git", "push", "origin", branch_name])


from datetime import datetime

@tool
def get_date(arg='today'):
    """Returns the current date as a string."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d")


# Home made tools
tools = [
    git_add, git_commit, git_push, get_date
]
# Add the file system toolkit tools
tools += toolkit.get_tools()


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, skilled at writing engaging blog posts and using the provided tools.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


llm_with_tools = llm.bind_tools(tools)



from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)


from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


query = """

You are an AI assistant with great technical knowledge that writes technical blog posts about implementing specific techniques in python.
You are an AI assistant with great knowledge about marketing and statistics that writes blog posts about practical statistical concepts for marketers.

You will be building a blog with a series of blog posts, which can be found in the _posts folder. 
It is a Jekyll blog hosted with GitHub pages. When I new post is commited and pushed to the gh-pages branch, it will be published on the blog.

Focus on writing helpful and engaging posts, like how-to posts, can increase the usefulness of the blog.
Always include code to illustrate techniques.
Avoid generic content/posts and buzz word content.
Then write the blog post and save it in the _posts folder as a markdown file with a suitable name.
Use the get date tool to get the current date, which you need for the blog post.  
The new blog post should increase the usefulness of the blog so that you can get more readers and build a broader audience.
Do not: write a blog post about exactly the same topic as an existing blog post.
Do not: write a blog post about a topic that is totally unrelated to the existing blog posts.
Do not: write a blog post that is a slight variation of a previous blog post.  
The blog post should be structured as follows:
A blog post always starts with a title, followed by a date (use get date tool), followed by category, followed by the actual blog post.    
-----

Example:
Title: Getting started with Jekyll
Date: 2024-08-26
Category: Blog  
etc.

-----

  

Always remember to use the write_file tool to write the blog post to the _posts folder as a markdown file.  

Write a new blog post with at least 1500 words.

Afterwards, use the provided tools to git add, commit and push (gh-pages branch) the new file to the github repository.

"""


agent_executor.invoke({"input": query})






