import json
import nltk
import requests
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import wikipediaapi
import os
import re
import ast
import operator

# Optional OpenAI integration: only used if `OPENAI_API_KEY` is set in env
try:
    import openai
except Exception:
    openai = None

# Optional sentence-transformers (BERT) for semantic intent matching
try:
    from sentence_transformers import SentenceTransformer as _STModel
    _SBERT = _STModel('all-MiniLM-L6-v2')
    _SBERT_AVAILABLE = True
    print('✅ Sentence-Transformers (BERT) loaded — semantic matching enabled.')
except Exception:
    _SBERT = None
    _SBERT_AVAILABLE = False
    print('ℹ️  sentence-transformers not installed — using TF-IDF only.')

# Hugging Face placeholder - we will call the Inference API when `HF_TOKEN` is set
HF_CHAT_API_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_HF_MODEL = os.environ.get('HF_MODEL', 'meta-llama/Llama-3.1-8B-Instruct')

# Try to load required NLTK data; if not present, it will be downloaded in app.py
lemmatizer = WordNetLemmatizer()

SAFE_MATH_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

SKILL_QUESTION_BANK = {
    'python': [
        'What are lists, tuples, sets, and dictionaries in Python?',
        'Explain the difference between a list and a tuple in Python.',
        'What are decorators in Python?',
        'How does exception handling work in Python?',
    ],
    'flask': [
        'What is Flask and why would you use it?',
        'How do routing, templates, and request handling work in Flask?',
        'How do you build and consume REST APIs in Flask?',
    ],
    'django': [
        'What is the difference between Django and Flask?',
        'How does Django ORM work?',
    ],
    'sql': [
        'What is the difference between INNER JOIN, LEFT JOIN, and RIGHT JOIN?',
        'What is normalization in databases?',
        'How do GROUP BY and HAVING work in SQL?',
    ],
    'mysql': [
        'How do you optimize slow MySQL queries?',
        'What indexes are and when would you use them in MySQL?',
    ],
    'html': [
        'What is the role of semantic HTML?',
        'What is the difference between block and inline elements?',
    ],
    'css': [
        'What is the difference between Flexbox and Grid?',
        'How does CSS specificity work?',
    ],
    'javascript': [
        'What is the difference between var, let, and const?',
        'Explain promises and async/await in JavaScript.',
    ],
    'react': [
        'What are props and state in React?',
        'What is the use of hooks such as useState and useEffect?',
    ],
    'pandas': [
        'What is a DataFrame in pandas?',
        'How do you handle missing values in pandas?',
    ],
    'numpy': [
        'What is the difference between a Python list and a NumPy array?',
        'Why is NumPy faster for numerical computation?',
    ],
    'machine learning': [
        'What is the difference between supervised and unsupervised learning?',
        'What is overfitting and how do you reduce it?',
        'How do you evaluate a machine learning model?',
    ],
    'deep learning': [
        'What is the difference between machine learning and deep learning?',
        'What is a neural network?',
        'What are activation functions in deep learning?',
    ],
    'data structures': [
        'What is the difference between a stack and a queue?',
        'When would you use an array versus a linked list?',
    ],
    'algorithms': [
        'What is time complexity and what does Big O notation mean?',
        'Explain the difference between linear search and binary search.',
    ],
    'git': [
        'What is the difference between git merge and git rebase?',
        'How do you resolve merge conflicts in Git?',
    ],
    'github': [
        'How do pull requests and code reviews work in GitHub?',
        'How do you collaborate on a shared repository using GitHub?',
    ],
}

GENERIC_PROJECT_QUESTIONS = [
    'Explain one project from your resume and your exact role in it.',
    'What problem was the project solving, and who were the users?',
    'What technology stack did you choose for the project and why?',
    'What was the biggest technical challenge in the project and how did you solve it?',
    'How did you design the backend, database, or API structure for the project?',
    'What improvements would you make if you had more time on that project?',
]

GENERIC_HR_QUESTIONS = [
    'Tell me about yourself.',
    'Walk me through your resume.',
    'Why do you want to work for this company?',
    'What are your strengths and weaknesses?',
    'Why should we hire you?',
    'Describe a challenge you faced and how you handled it.',
    'Where do you see yourself in the next 3 to 5 years?',
    'How do you handle pressure or tight deadlines?',
    'Describe a time you worked in a team.',
    'What motivates you in your career?',
]

LOCAL_KNOWLEDGE_BASE = {
    'pandas': 'Pandas is a Python library used for data analysis and data manipulation. It provides data structures like Series and DataFrame for working with tabular data efficiently.',
    'numpy': 'NumPy is a Python library for numerical computing. It provides fast array operations, mathematical functions, and tools for working with multidimensional data.',
    'python': 'Python is a high-level, easy-to-read programming language used for web development, automation, data science, machine learning, and many other tasks.',
    'flask': 'Flask is a lightweight Python web framework used to build web applications and APIs with simple routing, request handling, and template support.',
    'machine learning': 'Machine learning is a branch of artificial intelligence where systems learn patterns from data and make predictions or decisions without being explicitly programmed for every case.',
    'deep learning': 'Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn complex patterns from data such as images, text, and audio.',
    'large language model': 'A large language model, or LLM, is an AI model trained on large amounts of text data to understand and generate human-like language.',
    'data structures and algorithms': 'Data structures and algorithms are fundamental computer science concepts used to organize data efficiently and solve problems step by step.',
    'operating system': 'An operating system is system software that manages computer hardware, memory, files, and processes, and provides a platform for applications to run.',
    'array': 'An array is a data structure that stores multiple elements in an ordered sequence, usually allowing fast access by index.',
    'queue': 'A queue is a data structure that follows FIFO, first in first out, where the first element added is the first one removed.',
    'computer vision': 'Computer vision is a field of artificial intelligence that enables computers to interpret and analyze images and videos.',
    # --- CS fundamentals ---
    'data': 'Data is raw, unprocessed facts and figures such as numbers, text, images, or measurements. When organized and processed, data becomes information that can be used for analysis and decision-making.',
    'information': 'Information is processed, structured, or contextualized data that has meaning and is useful for decision-making. It answers questions like who, what, where, and when.',
    'data science': 'Data science is an interdisciplinary field that uses statistics, programming, and domain knowledge to extract insights and knowledge from structured and unstructured data.',
    'data mining': 'Data mining is the process of discovering patterns, correlations, and useful insights from large datasets using statistical, mathematical, and machine learning techniques.',
    'data analysis': 'Data analysis is the process of inspecting, cleaning, transforming, and modeling data to discover useful information, draw conclusions, and support decision-making.',
    'big data': 'Big data refers to extremely large datasets characterized by volume, velocity, and variety that cannot be processed by traditional tools. Technologies like Hadoop and Spark are used to handle it.',
    'data warehouse': 'A data warehouse is a centralized repository that stores large amounts of historical data from multiple sources, optimized for reporting and analytical queries.',
    'palindrome': 'A palindrome is a word, phrase, number, or sequence that reads the same forwards and backwards. Examples include "racecar", "level", "madam", and 121.',
    'anagram': 'An anagram is a word or phrase formed by rearranging the letters of another word or phrase using all original letters exactly once. For example, "listen" is an anagram of "silent".',
    'time complexity': 'Time complexity describes how the running time of an algorithm grows as the input size increases. It is expressed using Big O notation such as O(1) constant, O(n) linear, O(n log n) linearithmic, and O(n²) quadratic.',
    'space complexity': 'Space complexity measures how much memory an algorithm uses relative to the input size, also expressed in Big O notation.',
    'big o notation': 'Big O notation describes the upper bound of an algorithm\'s time or space complexity as input size grows. Common complexities: O(1) constant, O(log n) logarithmic, O(n) linear, O(n log n) linearithmic, O(n²) quadratic, O(2ⁿ) exponential.',
    'complexity': 'Complexity in algorithms refers to how resource usage (time or space) scales with input size, expressed in Big O notation.',
    'algorithm': 'An algorithm is a step-by-step procedure or formula for solving a problem. Good algorithms are correct, efficient, and terminating.',
    'recursion': 'Recursion is a programming technique where a function calls itself with a smaller input until it reaches a base case. It is commonly used in problems like factorial, Fibonacci, and tree traversal.',
    'dynamic programming': 'Dynamic programming solves complex problems by breaking them into overlapping subproblems and caching results to avoid redundant computation. Examples include Fibonacci, the knapsack problem, and shortest paths.',
    'greedy algorithm': 'A greedy algorithm makes the locally optimal choice at each step hoping to find a global optimum. Examples include Huffman coding, Dijkstra\'s algorithm, and the activity selection problem.',
    'memoization': 'Memoization is an optimization technique where results of expensive function calls are stored and reused when the same inputs appear again, speeding up recursive algorithms.',
    # --- Sorting and Searching ---
    'sorting': 'Sorting is the process of arranging elements in a particular order. Common algorithms: bubble sort O(n²), insertion sort O(n²), merge sort O(n log n), quick sort O(n log n) average.',
    'searching': 'Searching is the process of finding a specific element in a data structure. Linear search is O(n); binary search on sorted arrays is O(log n).',
    'binary search': 'Binary search finds a target in a sorted array by repeatedly halving the search space. It achieves O(log n) time complexity.',
    'linear search': 'Linear search checks each element one by one until the target is found. It has O(n) time complexity and works on unsorted data.',
    'bubble sort': 'Bubble sort repeatedly swaps adjacent elements that are in the wrong order. It has O(n²) average and worst-case time complexity.',
    'merge sort': 'Merge sort is a divide-and-conquer algorithm that splits an array in half, recursively sorts each half, and merges them. It guarantees O(n log n) time complexity.',
    'quick sort': 'Quick sort picks a pivot element and partitions the array around it. Its average time complexity is O(n log n), with O(n²) worst case.',
    'insertion sort': 'Insertion sort builds the sorted array one element at a time by inserting each element in its correct position. O(n²) worst case but efficient for small or nearly sorted data.',
    'selection sort': 'Selection sort repeatedly finds the minimum element from the unsorted portion and places it at the beginning. It has O(n²) time complexity.',
    # --- Data Structures ---
    'linked list': 'A linked list is a linear data structure where elements (nodes) contain data and a pointer to the next node. It allows efficient insertion and deletion but does not support random access.',
    'doubly linked list': 'A doubly linked list is a linked list where each node has pointers to both the next and previous nodes, enabling traversal in both directions.',
    'stack': 'A stack is a LIFO (last in, first out) data structure. Elements are pushed to and popped from the top. Used in function call management, undo operations, and expression evaluation.',
    'binary tree': 'A binary tree is a tree data structure where each node has at most two children (left and right). It is the basis for binary search trees, heaps, and expression trees.',
    'binary search tree': 'A BST (binary search tree) is a binary tree where each node\'s left subtree contains smaller values and the right subtree contains larger values, enabling O(log n) average search.',
    'heap': 'A heap is a complete binary tree satisfying the heap property: in a max-heap each parent ≥ its children; in a min-heap each parent ≤ its children. Used in priority queues and heap sort.',
    'hash table': 'A hash table maps keys to values using a hash function, giving O(1) average-case insert, delete, and lookup.',
    'hash map': 'A hash map stores key-value pairs and provides O(1) average-case access using a hash function to compute the index.',
    'graph': 'A graph is a data structure of nodes (vertices) connected by edges. Graphs can be directed or undirected and model networks, maps, and relationships.',
    'trie': 'A trie (prefix tree) stores strings character by character in a tree structure. It is efficient for autocomplete, spell checking, and prefix searches.',
    'priority queue': 'A priority queue is an abstract data type where each element has a priority. Elements are dequeued in priority order. Commonly implemented with a heap.',
    # --- OOP ---
    'object oriented programming': 'OOP organizes code around objects that bundle data (attributes) and behavior (methods). The four pillars are encapsulation, inheritance, polymorphism, and abstraction.',
    'oop': 'OOP (Object-Oriented Programming) is a paradigm based on objects with attributes and methods. The four pillars are encapsulation, inheritance, polymorphism, and abstraction.',
    'polymorphism': 'Polymorphism allows objects of different types to be treated through a common interface. It includes method overloading (compile-time) and method overriding (runtime).',
    'inheritance': 'Inheritance allows a child class to acquire properties and methods from a parent class, enabling code reuse and an "is-a" relationship.',
    'encapsulation': 'Encapsulation bundles data and methods in a class and restricts direct access using access modifiers (private, protected, public), protecting object state.',
    'abstraction': 'Abstraction hides complex implementation details and exposes only the necessary interface. It simplifies interactions and reduces coupling.',
    'class': 'A class is a blueprint for creating objects. It defines attributes (data) and methods (behavior) that its instances will have.',
    'object': 'An object is an instance of a class. It holds specific attribute values and can invoke the methods defined in its class.',
    # --- Python specifics ---
    'decorator': 'A decorator in Python wraps a function to add extra behavior without modifying it directly. Decorators use the @symbol syntax and are widely used in Flask routing.',
    'generator': 'A generator function yields values one at a time using the yield keyword, enabling lazy evaluation and memory-efficient iteration over large datasets.',
    'iterator': 'An iterator implements __iter__ and __next__ methods, allowing items to be traversed one at a time with a for loop or next().',
    'lambda': 'A lambda function is a small anonymous function defined with the lambda keyword. It can take multiple arguments but contains only a single expression.',
    'closure': 'A closure is a function that retains access to variables from its enclosing scope even after that scope has finished executing.',
    'list comprehension': 'List comprehension is a concise Python syntax to create lists: [expression for item in iterable if condition]. It is more readable and often faster than a for loop.',
    'exception handling': 'Exception handling uses try-except blocks to catch and handle runtime errors gracefully, preventing program crashes.',
    'scope': 'Scope defines where a variable is accessible. Python uses LEGB rule: Local, Enclosing, Global, and Built-in scopes.',
    'mutable': 'Mutable objects can be changed after creation. In Python, lists, dicts, and sets are mutable.',
    'immutable': 'Immutable objects cannot be changed after creation. In Python, strings, tuples, and integers are immutable.',
    'string': 'A string is a sequence of characters representing text. In Python, strings are immutable and support slicing, concatenation, formatting, and many built-in methods.',
    'list': 'A list in Python is an ordered, mutable collection that can hold items of different types, supporting indexing, slicing, and many built-in operations.',
    'tuple': 'A tuple in Python is an ordered, immutable collection. Unlike lists, tuples cannot be modified after creation.',
    'dictionary': 'A Python dictionary is an unordered collection of key-value pairs with O(1) average-case access, insertion, and deletion.',
    'set': 'A Python set is an unordered collection of unique elements supporting union, intersection, and difference operations.',
    'function': 'A function is a reusable block of code that performs a specific task, accepts optional parameters, and returns a result.',
    'variable': 'A variable is a named storage location that holds a value which can change during program execution.',
    'loop': 'A loop repeats a block of code multiple times. Python has for loops (iterating sequences) and while loops (repeating while a condition is true).',
    'conditional': 'Conditional statements execute code based on whether a condition is true or false. Python uses if, elif, and else.',
    # --- OS / Concurrency ---
    'deadlock': 'A deadlock occurs when two or more processes are each waiting for the other to release a resource, causing all of them to be stuck indefinitely. Prevention strategies include resource ordering and timeouts.',
    'race condition': 'A race condition occurs when multiple threads access shared data concurrently and the outcome depends on the timing of execution, leading to unpredictable results.',
    'mutex': 'A mutex (mutual exclusion lock) allows only one thread at a time to enter a critical section, preventing concurrent access to shared resources.',
    'semaphore': 'A semaphore is a synchronization primitive that controls access to shared resources using a counter that is atomically incremented (signal) or decremented (wait).',
    'process': 'A process is an instance of a program in execution with its own memory space and resources. Multiple processes can run concurrently.',
    'thread': 'A thread is the smallest unit of execution within a process. Threads share the same memory space and can run concurrently within a process.',
    'concurrency': 'Concurrency is the ability to handle multiple tasks at overlapping time periods by interleaving their execution on one or more processors.',
    'parallelism': 'Parallelism is the simultaneous execution of multiple tasks using multiple CPU cores. Unlike concurrency, tasks truly run at the same time.',
    'virtual memory': 'Virtual memory gives each process the illusion of a large private address space by mapping memory to disk, extending physical RAM.',
    'garbage collection': 'Garbage collection automatically reclaims memory that is no longer referenced by the program, preventing memory leaks.',
    'pointer': 'A pointer stores the memory address of another variable. Pointers are central to C/C++ and enable dynamic memory allocation and data structures like linked lists.',
    # --- Web / Networking ---
    'api': 'An API (Application Programming Interface) defines rules and protocols for software components to communicate with each other.',
    'rest api': 'A REST API uses HTTP methods (GET, POST, PUT, DELETE) to access and manipulate resources identified by URLs. Responses are typically JSON.',
    'http': 'HTTP (HyperText Transfer Protocol) is the protocol for transferring data on the web. Key methods are GET (read), POST (create), PUT (update), and DELETE.',
    'tcp ip': 'TCP/IP is the suite of internet communication protocols. TCP ensures reliable ordered delivery; IP handles addressing and routing.',
    'dns': 'DNS (Domain Name System) translates human-readable domain names like google.com into IP addresses that computers use to connect.',
    'cookie': 'A cookie is a small piece of data stored in the browser by a website to persist information like login state and preferences across sessions.',
    'session': 'A session stores user state on the server across multiple HTTP requests, identified by a session ID typically held in a cookie.',
    'jwt': 'JWT (JSON Web Token) is a compact token format for authentication, containing a header, payload, and signature encoded in Base64.',
    'oauth': 'OAuth is an authorization framework that lets third-party services access user accounts without exposing passwords, using access tokens.',
    'ssl tls': 'SSL/TLS are cryptographic protocols that encrypt data between client and server over the internet. TLS is the modern, secure successor to SSL.',
    'hashing': 'Hashing converts data of any size into a fixed-size hash value using a hash function. Used in data structures, password storage, and cryptography.',
    'encryption': 'Encryption converts plaintext into ciphertext using an algorithm and key so only authorized parties can decrypt and read it.',
    # --- Database ---
    'database': 'A database is an organized collection of data stored electronically. Relational databases use SQL; non-relational (NoSQL) databases use documents, key-value pairs, or graphs.',
    'normalization': 'Database normalization organizes tables to reduce redundancy and improve data integrity by eliminating duplicate data and structuring relationships.',
    'index': 'A database index is a data structure that speeds up data retrieval at the cost of extra storage. Without an index, a full table scan is needed.',
    'transaction': 'A database transaction is a sequence of operations treated as a single unit — either all succeed (commit) or all are rolled back.',
    'acid': 'ACID stands for Atomicity, Consistency, Isolation, and Durability — four properties guaranteeing reliable database transactions.',
    'nosql': 'NoSQL databases store data in formats other than relational tables — documents, key-value, graph, or wide-column. Examples: MongoDB, Redis, Cassandra.',
    'mongodb': 'MongoDB is a NoSQL document database that stores data as flexible JSON-like documents (BSON), known for scalability and schema flexibility.',
    'cache': 'A cache stores frequently accessed data in high-speed storage so future requests are served faster. Examples: CPU cache, Redis, browser cache.',
    # --- Tools and DevOps ---
    'git': 'Git is a distributed version control system for tracking source code changes. Developers use branches, commits, and merges to collaborate safely.',
    'docker': 'Docker packages applications and their dependencies into containers, ensuring consistent behavior across different environments.',
    'microservices': 'Microservices is an architectural style where an application is split into small, independently deployable services that communicate via APIs.',
    'cloud computing': 'Cloud computing delivers computing resources (servers, storage, databases) over the internet. Major providers include AWS, Azure, and Google Cloud.',
    'agile': 'Agile is a software development methodology emphasizing iterative development, collaboration, and response to change, organized in short sprints.',
    # --- Math / Logic ---
    'fibonacci': 'The Fibonacci sequence is a series where each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8, 13, 21 … Commonly used to demonstrate recursion and dynamic programming.',
    'factorial': 'The factorial of n (n!) is the product of all positive integers ≤ n. For example, 5! = 120. Computed recursively or iteratively; used in combinatorics.',
    'prime number': 'A prime number is a natural number greater than 1 with no divisors other than 1 and itself. Examples: 2, 3, 5, 7, 11, 13.',
    'number system': 'Number systems represent values in different bases: binary (base 2), octal (base 8), decimal (base 10), and hexadecimal (base 16).',
    'bit manipulation': 'Bit manipulation operates on integers at the binary level using bitwise operators: AND (&), OR (|), XOR (^), NOT (~), left shift (<<), and right shift (>>).',
    # --- Misc CS ---
    'compiler': 'A compiler translates high-level source code into machine code or bytecode before execution. Examples: GCC for C, javac for Java.',
    'interpreter': 'An interpreter executes source code line by line without pre-compiling to machine code. Python, Ruby, and JavaScript are interpreted.',
    'version control': 'Version control tracks changes to files over time so earlier versions can be restored. Git is the most widely used version control system.',
    'data type': 'A data type defines the kind of value a variable can hold (int, float, string, bool, list, etc.) and the operations permitted on it.',
    'sorting algorithm': 'Sorting algorithms arrange elements in order. Key ones: bubble sort O(n²), merge sort O(n log n), quick sort O(n log n) average, insertion sort O(n²).',
    'graph traversal': 'Graph traversal visits all nodes in a graph. BFS (breadth-first search) uses a queue and finds shortest paths; DFS (depth-first search) uses a stack or recursion.',
    'bfs': 'BFS (Breadth-First Search) traverses a graph level by level using a queue. It finds the shortest path in unweighted graphs and runs in O(V + E) time.',
    'dfs': 'DFS (Depth-First Search) traverses a graph by going as deep as possible along each branch before backtracking. It uses a stack or recursion and runs in O(V + E) time.',
    # --- AI / Neural Networks ---
    'neural network': 'A neural network is a computing system loosely inspired by the human brain. It consists of layers of interconnected nodes (neurons) — an input layer, one or more hidden layers, and an output layer. Each connection has a weight that adjusts during training so the network learns to map inputs to correct outputs.',
    'neural networks': 'Neural networks are computing systems loosely inspired by the human brain, made up of layers of interconnected nodes. They learn by adjusting connection weights during training and are the foundation of deep learning and modern AI applications like image recognition and language models.',
    'artificial intelligence': 'Artificial Intelligence (AI) is the simulation of human intelligence in machines. It includes learning, reasoning, and problem-solving. Examples: ChatGPT, self-driving cars, Alexa.',
    'natural language processing': 'Natural Language Processing (NLP) is a branch of AI that enables computers to understand, interpret, and generate human language. It powers chatbots, translation, sentiment analysis, and voice assistants.',
    'nlp': 'Natural Language Processing (NLP) is a branch of AI that enables computers to understand, interpret, and generate human language. It powers chatbots, translation, and voice assistants.',
    'reinforcement learning': 'Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment and receiving rewards or penalties, aiming to maximize total reward over time.',
    'transfer learning': 'Transfer learning reuses a pre-trained model on a new but related task, significantly reducing training time and data requirements. It is widely used in NLP (e.g. BERT, GPT) and computer vision.',
    'overfitting': 'Overfitting occurs when a machine learning model learns the training data too well, including noise, and performs poorly on new, unseen data. Techniques to reduce it include dropout, regularization, and cross-validation.',
    'supervised learning': 'Supervised learning trains a model on labeled input-output pairs so it can predict outputs for new inputs. Common algorithms: linear regression, decision trees, SVM, neural networks.',
    'unsupervised learning': 'Unsupervised learning finds hidden patterns in data without labeled examples. Common techniques: clustering (k-means), dimensionality reduction (PCA), and autoencoders.',
    # --- General Knowledge ---
    'capital of india': 'The capital of India is New Delhi. It is located in the northern part of the country and serves as the seat of the Indian government.',
    'capital of usa': 'The capital of the United States is Washington, D.C. It is the seat of the federal government and named after President George Washington.',
    'capital of uk': 'The capital of the United Kingdom is London. It is the largest city in the UK and home to the British Parliament and Buckingham Palace.',
    'capital of china': 'The capital of China is Beijing. It is the political, cultural, and educational centre of China.',
    'capital of france': 'The capital of France is Paris. It is known as the City of Light and is home to the Eiffel Tower and the Louvre.',
    'capital of japan': 'The capital of Japan is Tokyo. It is one of the most populous metropolitan areas in the world.',
    'capital of australia': 'The capital of Australia is Canberra. It was purpose-built as the capital and is located in the Australian Capital Territory.',
    'capital of canada': 'The capital of Canada is Ottawa, located in Ontario. It is home to the Parliament of Canada.',
    'capital of germany': 'The capital of Germany is Berlin. It is the largest city in Germany and the seat of the federal government.',
    'inventor of python': 'Python was created by Guido van Rossum and first released in 1991. He designed it to be easy to read and write, emphasizing code readability.',
    'who invented python': 'Python was invented by Guido van Rossum and first released in 1991.',
    'who invented java': 'Java was created by James Gosling at Sun Microsystems and first released in 1995.',
    'who invented c': 'The C programming language was created by Dennis Ritchie at Bell Labs in the early 1970s.',
    'software engineering': 'Software engineering is a branch of computer science and engineering focused on designing, developing, testing, and maintaining software systems in a structured and reliable way.',
    'internet': 'The internet is a global network of interconnected computers that communicate using standardized protocols (TCP/IP). It enables services like the World Wide Web, email, and messaging.',
    'world wide web': 'The World Wide Web (WWW) is a system of interlinked web pages and resources accessed via the internet using browsers. It was invented by Tim Berners-Lee in 1989.',
    'artificial neural network': 'An artificial neural network (ANN) is a model inspired by the human brain, composed of layers of connected nodes. It learns patterns from data by adjusting weights through training, and is used in classification, regression, and generative tasks.',
}

LOCAL_CODE_SNIPPETS = {
    'palindrome': (
        "def is_palindrome(s):\n"
        "    s = str(s)\n"
        "    return s == s[::-1]\n\n"
        "print(is_palindrome('racecar'))  # True\n"
        "print(is_palindrome('hello'))    # False"
    ),
    'anagram': (
        "def is_anagram(s1, s2):\n"
        "    return sorted(s1.lower()) == sorted(s2.lower())\n\n"
        "print(is_anagram('listen', 'silent'))  # True\n"
        "print(is_anagram('hello', 'world'))    # False"
    ),
    'fibonacci': (
        "def fibonacci(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        print(a, end=' ')\n"
        "        a, b = b, a + b\n\n"
        "fibonacci(10)  # 0 1 1 2 3 5 8 13 21 34"
    ),
    'factorial': (
        "def factorial(n):\n"
        "    if n == 0 or n == 1:\n"
        "        return 1\n"
        "    return n * factorial(n - 1)\n\n"
        "print(factorial(5))  # 120"
    ),
    'prime number': (
        "def is_prime(n):\n"
        "    if n < 2:\n"
        "        return False\n"
        "    for i in range(2, int(n ** 0.5) + 1):\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "    return True\n\n"
        "primes = [n for n in range(2, 50) if is_prime(n)]\n"
        "print(primes)"
    ),
    'bubble sort': (
        "def bubble_sort(arr):\n"
        "    n = len(arr)\n"
        "    for i in range(n):\n"
        "        for j in range(0, n - i - 1):\n"
        "            if arr[j] > arr[j + 1]:\n"
        "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
        "    return arr\n\n"
        "print(bubble_sort([64, 34, 25, 12, 22, 11, 90]))"
    ),
    'merge sort': (
        "def merge_sort(arr):\n"
        "    if len(arr) <= 1:\n"
        "        return arr\n"
        "    mid = len(arr) // 2\n"
        "    left = merge_sort(arr[:mid])\n"
        "    right = merge_sort(arr[mid:])\n"
        "    return merge(left, right)\n\n"
        "def merge(left, right):\n"
        "    result, i, j = [], 0, 0\n"
        "    while i < len(left) and j < len(right):\n"
        "        if left[i] <= right[j]:\n"
        "            result.append(left[i]); i += 1\n"
        "        else:\n"
        "            result.append(right[j]); j += 1\n"
        "    result.extend(left[i:])\n"
        "    result.extend(right[j:])\n"
        "    return result\n\n"
        "print(merge_sort([38, 27, 43, 3, 9, 82, 10]))"
    ),
    'quick sort': (
        "def quick_sort(arr):\n"
        "    if len(arr) <= 1:\n"
        "        return arr\n"
        "    pivot = arr[len(arr) // 2]\n"
        "    left   = [x for x in arr if x < pivot]\n"
        "    middle = [x for x in arr if x == pivot]\n"
        "    right  = [x for x in arr if x > pivot]\n"
        "    return quick_sort(left) + middle + quick_sort(right)\n\n"
        "print(quick_sort([3, 6, 8, 10, 1, 2, 1]))"
    ),
    'binary search': (
        "def binary_search(arr, target):\n"
        "    left, right = 0, len(arr) - 1\n"
        "    while left <= right:\n"
        "        mid = (left + right) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            left = mid + 1\n"
        "        else:\n"
        "            right = mid - 1\n"
        "    return -1\n\n"
        "arr = [1, 3, 5, 7, 9, 11]\n"
        "print(binary_search(arr, 7))  # 3"
    ),
    'linked list': (
        "class Node:\n"
        "    def __init__(self, data):\n"
        "        self.data = data\n"
        "        self.next = None\n\n"
        "class LinkedList:\n"
        "    def __init__(self):\n"
        "        self.head = None\n\n"
        "    def append(self, data):\n"
        "        new_node = Node(data)\n"
        "        if not self.head:\n"
        "            self.head = new_node; return\n"
        "        curr = self.head\n"
        "        while curr.next:\n"
        "            curr = curr.next\n"
        "        curr.next = new_node\n\n"
        "    def display(self):\n"
        "        curr = self.head\n"
        "        while curr:\n"
        "            print(curr.data, end=' -> ')\n"
        "            curr = curr.next\n"
        "        print('None')\n\n"
        "ll = LinkedList()\n"
        "ll.append(1); ll.append(2); ll.append(3)\n"
        "ll.display()  # 1 -> 2 -> 3 -> None"
    ),
    'stack': (
        "class Stack:\n"
        "    def __init__(self):\n"
        "        self.items = []\n\n"
        "    def push(self, item): self.items.append(item)\n"
        "    def pop(self): return self.items.pop() if self.items else None\n"
        "    def peek(self): return self.items[-1] if self.items else None\n"
        "    def is_empty(self): return len(self.items) == 0\n\n"
        "s = Stack()\n"
        "s.push(1); s.push(2); s.push(3)\n"
        "print(s.pop())   # 3\n"
        "print(s.peek())  # 2"
    ),
    'queue': (
        "from collections import deque\n\n"
        "class Queue:\n"
        "    def __init__(self):\n"
        "        self.items = deque()\n\n"
        "    def enqueue(self, item): self.items.append(item)\n"
        "    def dequeue(self): return self.items.popleft() if self.items else None\n"
        "    def is_empty(self): return len(self.items) == 0\n\n"
        "q = Queue()\n"
        "q.enqueue(1); q.enqueue(2)\n"
        "print(q.dequeue())  # 1"
    ),
    'binary tree': (
        "class TreeNode:\n"
        "    def __init__(self, val):\n"
        "        self.val = val\n"
        "        self.left = self.right = None\n\n"
        "def inorder(root):\n"
        "    if root:\n"
        "        inorder(root.left)\n"
        "        print(root.val, end=' ')\n"
        "        inorder(root.right)\n\n"
        "root = TreeNode(4)\n"
        "root.left = TreeNode(2); root.right = TreeNode(6)\n"
        "root.left.left = TreeNode(1); root.left.right = TreeNode(3)\n"
        "inorder(root)  # 1 2 3 4 6"
    ),
    'reverse string': (
        "def reverse_string(s):\n"
        "    return s[::-1]\n\n"
        "print(reverse_string('hello'))  # olleh"
    ),
    'reverse linked list': (
        "class Node:\n"
        "    def __init__(self, data):\n"
        "        self.data = data; self.next = None\n\n"
        "def reverse(head):\n"
        "    prev, curr = None, head\n"
        "    while curr:\n"
        "        nxt = curr.next\n"
        "        curr.next = prev\n"
        "        prev = curr\n"
        "        curr = nxt\n"
        "    return prev"
    ),
    'bfs': (
        "from collections import deque\n\n"
        "def bfs(graph, start):\n"
        "    visited = set()\n"
        "    queue = deque([start])\n"
        "    visited.add(start)\n"
        "    while queue:\n"
        "        node = queue.popleft()\n"
        "        print(node, end=' ')\n"
        "        for neighbor in graph[node]:\n"
        "            if neighbor not in visited:\n"
        "                visited.add(neighbor)\n"
        "                queue.append(neighbor)\n\n"
        "graph = {0: [1, 2], 1: [3], 2: [4], 3: [], 4: []}\n"
        "bfs(graph, 0)  # 0 1 2 3 4"
    ),
    'dfs': (
        "def dfs(graph, node, visited=None):\n"
        "    if visited is None:\n"
        "        visited = set()\n"
        "    visited.add(node)\n"
        "    print(node, end=' ')\n"
        "    for neighbor in graph[node]:\n"
        "        if neighbor not in visited:\n"
        "            dfs(graph, neighbor, visited)\n\n"
        "graph = {0: [1, 2], 1: [3], 2: [4], 3: [], 4: []}\n"
        "dfs(graph, 0)  # 0 1 3 2 4"
    ),
    'calculator': (
        "def calculator(a, op, b):\n"
        "    if op == '+': return a + b\n"
        "    if op == '-': return a - b\n"
        "    if op == '*': return a * b\n"
        "    if op == '/':\n"
        "        if b == 0: return 'Error: division by zero'\n"
        "        return a / b\n"
        "    return 'Unknown operator'\n\n"
        "print(calculator(10, '+', 5))  # 15\n"
        "print(calculator(10, '/', 2))  # 5.0"
    ),
}

class ChatBot:
    def __init__(self, intents_file='intents.json'):
        self.intents_file = intents_file
        self.intents = []
        self.patterns = []
        self.responses = []
        self.tags = []
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenize_and_lemmatize, stop_words='english')
        self.tfidf_matrix = None
        self._load_intents()
        self._train_model()
        
        # Initialize Wikipedia API client (no auth required)
        self.wiki = wikipediaapi.Wikipedia(user_agent='PyChatbot/1.0 (educational-project)', language='en')
        
    def _load_intents(self):
        """Loads the intents from the JSON file."""
        with open(self.intents_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for intent in data['intents']:
            for pattern in intent['patterns']:
                self.patterns.append(pattern)
                self.responses.append(intent['responses'])
                self.tags.append(intent['tag'])
                self.intents.append(intent)

    def tokenize_and_lemmatize(self, text):
        """Tokenizes and lemmatizes the input text."""
        try:
            tokens = nltk.word_tokenize(text.lower())
        except LookupError:
            tokens = text.lower().split()
        return [lemmatizer.lemmatize(word) for word in tokens]

    def _is_knowledge_query(self, text):
        """Heuristic to detect factual/general queries that should prefer generative fallback."""
        normalized = text.strip().lower()
        starters = (
            'what is', 'who is', 'where is', 'when is', 'why', 'how',
            'what does', 'what are', 'can you explain', 'define', 'explain', 'tell me about', 'name one', 'list',
            'difference between', 'compare', 'application of', 'applications of'
        )
        return normalized.endswith('?') or normalized.startswith(starters)

    def _split_compound_questions(self, text):
        """Split a message containing multiple questions into smaller prompts."""
        normalized = re.sub(r'\s+', ' ', text.strip())
        if not normalized:
            return []

        parts = re.split(
            r'(?<=[?.!])\s+(?=(?:what|how|why|who|where|when|define|explain|tell|write|name|compare|list|can)\b)',
            normalized,
            flags=re.IGNORECASE,
        )

        questions = []
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                questions.append(cleaned)
        return questions

    def _extract_quoted_prompts(self, text):
        """Extract multiple quoted prompts from a single message."""
        matches = re.findall(r'"([^"]+)"', text)
        cleaned = [match.strip() for match in matches if match.strip()]
        return cleaned if len(cleaned) > 1 else []

    def _safe_eval_math(self, expression):
        """Safely evaluate a small arithmetic expression."""
        def _eval_node(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in SAFE_MATH_OPERATORS:
                return SAFE_MATH_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_MATH_OPERATORS:
                return SAFE_MATH_OPERATORS[type(node.op)](_eval_node(node.operand))
            raise ValueError('Unsupported expression')

        try:
            parsed = ast.parse(expression, mode='eval')
        except SyntaxError:
            raise ValueError('Invalid math expression')
        return _eval_node(parsed.body)

    def _get_math_response(self, text):
        """Return an answer for simple arithmetic questions if present."""
        normalized = text.strip().lower().rstrip('?.!')
        normalized = re.sub(r'^(what is)(?:\s+the\s+value\s+of)?\s+', 'what is ', normalized)
        match = re.search(r'(?:what is|calculate|solve|evaluate)\s+([-+*/%()\d\s.]+)$', normalized)
        if not match:
            # Also handle bare arithmetic expressions like "2-3", "5 * 3", "10 / 2"
            bare = normalized.strip()
            if re.fullmatch(r'[-+*/%()\d\s.]+', bare) and re.search(r'[+\-*/%(]', bare):
                expression = bare
            else:
                return None
        else:
            expression = match.group(1).strip()

        if not re.fullmatch(r'[-+*/%()\d\s.]+', expression):
            return None

        try:
            result = self._safe_eval_math(expression)
        except Exception:
            return None

        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"The answer is {result}."

    def _build_model_messages(self, user_input, context_text=None, context_name=None, chat_history=None):
        """Build chat messages for generative backends, including conversation history."""
        system_content = (
            "You are a helpful assistant. Answer the exact user question clearly and concisely. "
            "If a term has multiple meanings, use the user's context."
        )

        messages = [{"role": "system", "content": system_content}]

        # Include the last 4 conversation turns (8 messages) for context memory
        if chat_history:
            for turn in chat_history[-8:]:
                messages.append({"role": turn["role"], "content": turn["content"]})

        if context_text:
            excerpt = context_text[:12000]
            user_content = (
                f"Use the uploaded file content to answer the question. "
                f"If the answer is not in the file, say that briefly.\n\n"
                f"File: {context_name or 'uploaded file'}\n"
                f"Content:\n{excerpt}\n\n"
                f"Question: {user_input}"
            )
        else:
            user_content = user_input

        messages.append({"role": "user", "content": user_content})
        return messages

    def _extract_candidate_name(self, context_text):
        """Best-effort name extraction from uploaded resume text."""
        lines = [line.strip() for line in context_text.splitlines() if line.strip()]
        for line in lines[:8]:
            if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}", line):
                return line
        return None

    def _extract_skills_from_context(self, context_text):
        """Extract known technical skills from uploaded text using a curated vocabulary."""
        lowered = context_text.lower()
        ordered_skills = []
        for skill in SKILL_QUESTION_BANK:
            if skill in lowered:
                ordered_skills.append(skill)
        return ordered_skills

    def _extract_project_lines(self, context_text):
        """Extract probable project titles or project bullets from resume text."""
        lines = [line.strip(' -:\t') for line in context_text.splitlines() if line.strip()]
        project_lines = []
        in_project_section = False

        for line in lines:
            lowered = line.lower()
            if any(keyword in lowered for keyword in ('project', 'projects', 'academic project', 'personal project')):
                in_project_section = True
                if len(line.split()) > 1 and 'project' not in lowered[:12]:
                    project_lines.append(line)
                continue

            if in_project_section and re.fullmatch(r'[A-Z][A-Za-z\s]+', line) and len(line.split()) <= 6:
                project_lines.append(line)
                continue

            if in_project_section and len(project_lines) < 4 and len(line.split()) <= 12:
                project_lines.append(line)

            if in_project_section and len(project_lines) >= 4:
                break

        unique_lines = []
        seen = set()
        for line in project_lines:
            normalized = line.lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_lines.append(line)
        return unique_lines[:4]

    def _get_contextual_file_response(self, user_input, context_text, context_name=None):
        """Answer common uploaded-file questions without relying on external models."""
        normalized = user_input.strip().lower()
        candidate_name = self._extract_candidate_name(context_text)
        skills = self._extract_skills_from_context(context_text)
        asks_for_interview_questions = 'interview' in normalized and any(
            term in normalized for term in ('question', 'questions', 'cv', 'resume', 'uploaded file', 'possible')
        )
        asks_for_technical_questions = 'technical question' in normalized or 'technical questions' in normalized
        asks_for_project_questions = 'project' in normalized and 'question' in normalized
        asks_for_hr_questions = 'hr' in normalized and 'question' in normalized

        if asks_for_technical_questions or asks_for_interview_questions:
            questions = []
            for skill in skills[:6]:
                questions.extend(SKILL_QUESTION_BANK.get(skill, []))

            if asks_for_interview_questions:
                questions.extend(GENERIC_PROJECT_QUESTIONS)

            if not questions:
                questions = [
                    'Tell me about yourself and your recent projects.',
                    'What technical skills are you most confident in?',
                    'Explain one project you built and the challenges you solved.',
                    'How do you debug a problem in your code?',
                    'What have you learned recently that improved your development skills?',
                ]

            unique_questions = []
            seen = set()
            for question in questions:
                if question not in seen:
                    seen.add(question)
                    unique_questions.append(question)

            intro = 'Here are likely technical interview questions based on the uploaded CV:'
            if candidate_name:
                intro = f'Here are likely technical interview questions for {candidate_name} based on the uploaded CV:'

            lines = [intro]
            for index, question in enumerate(unique_questions[:15], start=1):
                lines.append(f'{index}. {question}')
            return '\n'.join(lines)

        if asks_for_project_questions:
            project_lines = self._extract_project_lines(context_text)
            questions = list(GENERIC_PROJECT_QUESTIONS)

            for skill in skills[:5]:
                questions.append(f'How did you use {skill.title()} in your project work?')

            for project in project_lines:
                questions.append(f'Explain the project "{project}" in detail.')
                questions.append(f'What challenges did you face while building "{project}"?')

            unique_questions = []
            seen = set()
            for question in questions:
                if question not in seen:
                    seen.add(question)
                    unique_questions.append(question)

            intro = 'Here are project-related interview questions based on the uploaded CV:'
            if candidate_name:
                intro = f'Here are project-related interview questions for {candidate_name} based on the uploaded CV:'

            lines = [intro]
            for index, question in enumerate(unique_questions[:15], start=1):
                lines.append(f'{index}. {question}')
            return '\n'.join(lines)

        if asks_for_hr_questions:
            questions = list(GENERIC_HR_QUESTIONS)
            if candidate_name:
                questions.insert(1, f'Introduce yourself as {candidate_name} in a concise and professional way.')

            if skills:
                primary_skills = ', '.join(skill.title() for skill in skills[:5])
                questions.append(f'How would you explain your strongest skills: {primary_skills}?')

            project_lines = self._extract_project_lines(context_text)
            for project in project_lines[:2]:
                questions.append(f'Which project are you most proud of, such as "{project}", and why?')

            unique_questions = []
            seen = set()
            for question in questions:
                if question not in seen:
                    seen.add(question)
                    unique_questions.append(question)

            intro = 'Here are HR interview questions based on the uploaded CV:'
            if candidate_name:
                intro = f'Here are HR interview questions for {candidate_name} based on the uploaded CV:'

            lines = [intro]
            for index, question in enumerate(unique_questions[:15], start=1):
                lines.append(f'{index}. {question}')
            return '\n'.join(lines)

        if 'skill' in normalized or 'technology' in normalized:
            if skills:
                formatted = ', '.join(skill.title() for skill in skills[:12])
                return f'The uploaded file mentions these technical skills: {formatted}.'
            return 'I could not confidently identify technical skills in the uploaded file.'

        if 'name' in normalized and candidate_name:
            return f'The candidate name in the uploaded file is {candidate_name}.'

        if 'summary' in normalized or 'summarize' in normalized:
            if skills:
                skill_summary = ', '.join(skill.title() for skill in skills[:8])
                if candidate_name:
                    return f'{candidate_name} appears to have experience with: {skill_summary}.'
                return f'The uploaded file highlights experience with: {skill_summary}.'

        return None

    def _has_extra_topic_tokens(self, user_input, matched_pattern):
        """Detect when a question includes topic words not present in the matched canned pattern."""
        ignore_tokens = {
            'what', 'is', 'are', 'who', 'where', 'when', 'why', 'how',
            'define', 'explain', 'tell', 'me', 'about', 'the', 'a', 'an',
            'in', 'of', 'to', 'for', 'language', '?'
        }
        user_tokens = {t for t in self.tokenize_and_lemmatize(user_input) if t not in ignore_tokens}
        pattern_tokens = {t for t in self.tokenize_and_lemmatize(matched_pattern) if t not in ignore_tokens}
        return len(user_tokens - pattern_tokens) > 0

    def _train_model(self):
        """Fits the TF-IDF vectorizer and precomputes BERT sentence embeddings (if available)."""
        self.tfidf_matrix = self.vectorizer.fit_transform(self.patterns)
        if _SBERT_AVAILABLE:
            self.pattern_embeddings = _SBERT.encode(self.patterns, show_progress_bar=False)
        else:
            self.pattern_embeddings = None

    def _normalize_user_input(self, text):
        """Normalize common abbreviations and typos before matching/fallback."""
        normalized = text.strip()
        normalized = re.sub(r'^\s*waht\b', 'what', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bdeep\s+lern(ing)?\b', 'deep learning', normalized, flags=re.IGNORECASE)

        typo_replacements = {
            r'\bquueu\b': 'queue',
            r'\bqeue\b': 'queue',
            r'\barrary\b': 'array',
            r'\balogrithm\b': 'algorithm',
            r'\boperting\b': 'operating',
        }
        for pattern, replacement in typo_replacements.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        # Expand common technical abbreviations in question-style queries.
        abbreviation_replacements = {
            r'\bdl\b': 'deep learning',
            r'\bllm\b': 'large language model',
            r'\bdsa\b': 'data structures and algorithms',
            r'\bos\b': 'operating system',
            r'\bml\b': 'machine learning',
            r'\boops\b': 'object oriented programming',
        }
        if re.search(r'\b(what is|define|explain|about|tell me about)\b', normalized, flags=re.IGNORECASE):
            for pattern, replacement in abbreviation_replacements.items():
                normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        return normalized

    def _get_local_knowledge_response(self, user_input):
        """Return a deterministic answer for common technical concepts."""
        normalized = self._normalize_user_input(user_input).strip().lower()

        # Strip every common conversational / question-starter prefix so the remaining
        # string is the bare topic.  Order matters: longer patterns first.
        query_patterns = (
            # e.g. "can you please explain me what is data structure"
            r'^(?:can you (?:please )?(?:explain|tell) (?:me )?(?:about |what is |what are )?)',
            r'^(?:could you (?:please )?(?:explain|tell) (?:me )?(?:about |what is |what are )?)',
            r'^(?:please (?:explain|define|tell me about|describe)\s+)',
            r'^(?:i want to know (?:about |what is |what are )?)',
            r'^(?:tell me (?:about |more about |what is |what are )?)',
            r'^(?:explain (?:me )?(?:what is |what are |about )?)',
            r'^(?:what is|what are|define|describe|tell me about)\s+',
            r'^(?:name one application of|applications of)\s+',
        )
        topic = normalized
        for pattern in query_patterns:
            topic = re.sub(pattern, '', topic, flags=re.IGNORECASE).strip()

        topic = topic.rstrip('?.! ')
        # Strip trailing qualifiers like "in simple words", "briefly", "with example"
        # so "neural networks in simple words" → "neural networks" for correct KB lookup
        topic = re.sub(
            r'\s+(?:in simple words?|in brief|briefly|with examples?|with an example|'
            r'for beginners?|easily|simply|in detail|in short|in layman terms?|'
            r'in easy words?|step by step|clearly)$',
            '', topic, flags=re.IGNORECASE
        ).strip()
        topic = re.sub(r'\bin python\b', '', topic, flags=re.IGNORECASE).strip()

        alias_map = {
            'llm': 'large language model',
            'dsa': 'data structures and algorithms',
            'os': 'operating system',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'oops': 'object oriented programming',
            'oop': 'object oriented programming',
            'bst': 'binary search tree',
            'big o': 'big o notation',
            'big o notation': 'big o notation',
            'time complexity': 'time complexity',
            'space complexity': 'space complexity',
            'deadlock': 'deadlock',
            'race condition': 'race condition',
            'linked list': 'linked list',
            'doubly linked list': 'doubly linked list',
            'hash map': 'hash map',
            'hash table': 'hash table',
            'rest': 'rest api',
            'rest api': 'rest api',
            'tcp': 'tcp ip',
            'tcp/ip': 'tcp ip',
            'ssl': 'ssl tls',
            'tls': 'ssl tls',
            'ssl/tls': 'ssl tls',
        }
        topic = alias_map.get(topic, topic)

        if topic in LOCAL_KNOWLEDGE_BASE:
            return LOCAL_KNOWLEDGE_BASE[topic]

        # Pass 1: a stored key phrase is a substring of the user topic — longest key wins.
        # Skip keys where the topic merely *starts with* that key followed by more words,
        # because pass 2 may find a longer, more specific key (e.g. "data structures and
        # algorithms") that should beat the shorter "data" key for topic "data structures".
        for key in sorted(LOCAL_KNOWLEDGE_BASE, key=len, reverse=True):
            if key in topic and len(key) >= 4:
                if topic.startswith(key) and topic != key:
                    continue  # let pass 2 find a longer matching key
                return LOCAL_KNOWLEDGE_BASE[key]

        # Pass 2: user topic is a prefix/substring of a longer key — handles queries like
        # "data structures" → "data structures and algorithms". Require len >= 6 to avoid
        # common short words triggering false matches.
        if len(topic) >= 6:
            for key in sorted(LOCAL_KNOWLEDGE_BASE, key=len, reverse=True):
                if topic in key:
                    return LOCAL_KNOWLEDGE_BASE[key]

        return None

    def _get_code_response(self, user_input):
        """Return a code snippet if the user asks to write/show/implement something."""
        normalized = user_input.strip().lower().rstrip('?.!')
        code_triggers = (
            'write a', 'write the', 'write code', 'write program', 'write a program',
            'give code', 'give me code', 'show code', 'show me code',
            'code for', 'code to', 'code of',
            'program for', 'program to', 'program of',
            'implement', 'implementation of', 'implementation for',
            'example of', 'example for',
            'how to check', 'how to find', 'how to detect',
            'check code', 'check program',
        )
        is_code_request = any(trigger in normalized for trigger in code_triggers)
        if not is_code_request:
            return None

        for key, snippet in LOCAL_CODE_SNIPPETS.items():
            if key in normalized:
                return f'Here is a Python implementation of {key}:\n\n```python\n{snippet}\n```'

        return None

    def _should_use_context_for_query(self, user_input):
        """Use uploaded file context only when the question clearly refers to that file."""
        normalized = user_input.strip().lower()

        file_reference_terms = (
            'uploaded file', 'this file', 'the file', 'from the file', 'in the file',
            'resume', 'my resume', 'cv', 'my cv', 'document', 'pdf', 'attachment',
            'uploaded pdf', 'uploaded resume'
        )
        if any(term in normalized for term in file_reference_terms):
            return True

        context_question_terms = (
            'technical question', 'technical questions', 'interview question', 'interview questions',
            'project related', 'project questions', 'hr question', 'hr questions',
            'skills', 'skill', 'technology', 'technologies', 'candidate name', 'summary', 'summarize'
        )
        if any(term in normalized for term in context_question_terms):
            return True

        return False

    def get_generative_response(self, user_input, context_text=None, context_name=None, chat_history=None):
        """Uses Wikipedia's free API (no auth needed) to answer general knowledge questions."""
        local_knowledge_response = self._get_local_knowledge_response(user_input)
        if local_knowledge_response:
            return local_knowledge_response

        if context_text:
            local_response = self._get_contextual_file_response(user_input, context_text, context_name=context_name)
            if local_response:
                return local_response

            messages = self._build_model_messages(user_input, context_text=context_text, context_name=context_name, chat_history=chat_history)

            if openai and os.environ.get('OPENAI_API_KEY'):
                try:
                    openai.api_key = os.environ.get('OPENAI_API_KEY')
                    resp = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        max_tokens=350,
                        temperature=0.2,
                    )
                    return resp.choices[0].message.content.strip()
                except Exception as oe:
                    print(f"OpenAI contextual fallback error: {oe}")

            hf_token = os.environ.get('HF_TOKEN')
            if hf_token:
                try:
                    model = os.environ.get('HF_MODEL', DEFAULT_HF_MODEL)
                    hf_headers = {"Authorization": f"Bearer {hf_token}", "Accept": "application/json"}
                    payload = {
                        "model": model,
                        "messages": messages,
                        "max_tokens": 320,
                        "temperature": 0.2
                    }
                    hf_resp = requests.post(HF_CHAT_API_URL, headers=hf_headers, json=payload, timeout=60)
                    if hf_resp.status_code == 200:
                        data = hf_resp.json()
                        choices = data.get('choices', []) if isinstance(data, dict) else []
                        if choices and isinstance(choices[0], dict):
                            content = choices[0].get('message', {}).get('content')
                            if content:
                                return content.strip()
                    else:
                        print(f"HuggingFace contextual chat API returned {hf_resp.status_code}: {hf_resp.text}")
                except Exception as he:
                    print(f"HuggingFace contextual fallback error: {he}")

            return "I uploaded the file, but I could not generate an answer from it right now."

        try:
            # Clean the query for better Wikipedia search:
            # 1. Strip conversational starters so "what is software engineering" → "software engineering"
            # 2. Strip trailing qualifiers like "in simple words", "with example", "briefly"
            # 3. Remove code/programming noise words
            clean_starters = r'^(?:what is|what are|what does|who is|who invented|who created|define|explain|tell me about|how does|how do|give me|show me)\s+'
            trailing_qualifiers = r'\s+(?:in simple words?|in brief|briefly|with examples?|with an example|for beginners?|easily|simply|in detail|in short)$'
            noise_words = r'\b(code|program|write|example)\b'
            cleaned_input = re.sub(clean_starters, '', user_input, flags=re.IGNORECASE)
            cleaned_input = re.sub(trailing_qualifiers, '', cleaned_input, flags=re.IGNORECASE)
            cleaned_input = re.sub(noise_words, '', cleaned_input, flags=re.IGNORECASE).strip()
            search_query = cleaned_input if len(cleaned_input) > 2 else user_input
            print(f"Fallback triggered. Searching Wikipedia for: '{search_query}' (original: '{user_input}')")
            
            # Step 1: Use Wikipedia OpenSearch to find the best matching article title
            headers = {"User-Agent": "PyChatbot/1.0 (educational-project; contact@example.com)"}
            search_res = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": search_query,
                    "limit": 1,
                    "namespace": 0,
                    "format": "json"
                },
                headers=headers,
                timeout=8
            )
            search_data = search_res.json()
            
            # Step 2: Fetch article summary if a match was found
            if len(search_data) >= 2 and len(search_data[1]) > 0:
                article_title = search_data[1][0]
                print(f"Best Wikipedia match: '{article_title}'")
                page = self.wiki.page(article_title)
                
                if page.exists() and page.summary:
                    # Return first 3 sentences for a concise, natural answer
                    sentences = page.summary.split('. ')
                    short_answer = '. '.join(sentences[:3]).strip()
                    if not short_answer.endswith('.'):
                        short_answer += '.'
                    return f"📚 {short_answer}"

        except Exception as e:
            print(f"Wikipedia fallback error: {e}")

        # If Wikipedia has no match or fails, try OpenAI (if available)
        if openai and os.environ.get('OPENAI_API_KEY'):
            try:
                openai.api_key = os.environ.get('OPENAI_API_KEY')
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=self._build_model_messages(user_input, chat_history=chat_history),
                    max_tokens=250,
                    temperature=0.2,
                )
                text = resp.choices[0].message.content.strip()
                return text
            except Exception as oe:
                print(f"OpenAI fallback error: {oe}")

        # Next fallback: Hugging Face chat-completions API (if HF_TOKEN available)
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            try:
                model = os.environ.get('HF_MODEL', DEFAULT_HF_MODEL)
                hf_headers = {"Authorization": f"Bearer {hf_token}", "Accept": "application/json"}
                payload = {
                    "model": model,
                    "messages": self._build_model_messages(user_input, chat_history=chat_history),
                    "max_tokens": 220,
                    "temperature": 0.2
                }
                hf_resp = requests.post(HF_CHAT_API_URL, headers=hf_headers, json=payload, timeout=60)
                if hf_resp.status_code == 200:
                    data = hf_resp.json()
                    choices = data.get('choices', []) if isinstance(data, dict) else []
                    if choices and isinstance(choices[0], dict):
                        message = choices[0].get('message', {})
                        content = message.get('content')
                        if content:
                            return content.strip()
                else:
                    print(f"HuggingFace chat API returned {hf_resp.status_code}: {hf_resp.text}")
            except Exception as he:
                print(f"HuggingFace fallback error: {he}")

        return "I wasn't able to find specific information on that. Could you try rephrasing your question?"

    def _get_single_response(self, user_input, context_text=None, context_name=None, chat_history=None):
        """Return a response for a single prompt/question."""
        # Detect follow-up phrases like "explain more", "tell me more", "give an example".
        # kb_input  — used for local KB / TF-IDF (reuses prior topic so KB matching works)
        # llm_input — sent to LLM generative path with full context for richer answers
        kb_input  = user_input
        llm_input = user_input
        follow_up_words = {'it', 'this', 'that', 'they', 'more', 'elaborate', 'details', 'example'}
        if chat_history and len(user_input.split()) <= 6 and set(user_input.lower().split()) & follow_up_words:
            last_user_q = next((m['content'] for m in reversed(chat_history) if m['role'] == 'user'), None)
            if last_user_q:
                kb_input  = last_user_q                           # prior topic for local lookups
                llm_input = f"{last_user_q} — {user_input}"      # full context for LLM

        math_response = self._get_math_response(user_input)
        if math_response:
            return math_response

        code_response = self._get_code_response(kb_input)
        if code_response:
            return code_response

        local_kb_response = self._get_local_knowledge_response(kb_input)
        if local_kb_response:
            return local_kb_response

        if context_text and self._should_use_context_for_query(kb_input):
            return self.get_generative_response(llm_input, context_text=context_text, context_name=context_name, chat_history=chat_history)

        normalized_input = self._normalize_user_input(kb_input)
        user_tfidf = self.vectorizer.transform([normalized_input])
        cosine_similarities = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()
        best_match_idx = int(np.argmax(cosine_similarities))
        best_score = float(cosine_similarities[best_match_idx])
        best_tag = self.tags[best_match_idx]

        # BERT semantic similarity override — helps with paraphrases TF-IDF misses
        if _SBERT_AVAILABLE and self.pattern_embeddings is not None:
            user_embed = _SBERT.encode([normalized_input])
            sbert_sims = cosine_similarity(user_embed, self.pattern_embeddings).flatten()
            sbert_best_idx = int(np.argmax(sbert_sims))
            sbert_best_score = float(sbert_sims[sbert_best_idx])
            if sbert_best_score > best_score + 0.08:
                print(f'SBERT override: {self.tags[sbert_best_idx]} ({sbert_best_score:.4f}) > TF-IDF {best_tag} ({best_score:.4f})')
                best_match_idx = sbert_best_idx
                best_score = sbert_best_score
                best_tag = self.tags[best_match_idx]
        
        # Log the similarity score for debugging
        print(f"Input: '{user_input}' | Normalized: '{normalized_input}' | Match: '{best_tag}' | Score: {best_score:.4f}")
        
        # Only use intent-based response if similarity is above a strict threshold
        small_talk_tags = {
            'greeting', 'goodbye', 'thanks', 'about', 'capabilities', 'project_details', 'joke',
            'bot_identity', 'bot_capabilities_food', 'philosophical', 'impossible_hypothetical',
            'emotions_abstract', 'general_chitchat', 'compliment', 'insult',
        }
        # Conversational / philosophical intents get a lower threshold — they don't need
        # high lexical overlap to be valid (e.g. "can you eat food?" is clearly a bot-capability question)
        conversational_tags = {
            'bot_identity', 'bot_capabilities_food', 'philosophical', 'impossible_hypothetical',
            'emotions_abstract', 'general_chitchat', 'compliment', 'insult',
            'greeting', 'goodbye', 'thanks', 'joke',
        }
        is_knowledge_query = self._is_knowledge_query(normalized_input)
        matched_pattern = self.patterns[best_match_idx]
        has_extra_topic_tokens = self._has_extra_topic_tokens(normalized_input, matched_pattern)

        # Conversational/philosophical intents always respond directly — never send to generative fallback
        if best_tag in conversational_tags and best_score > 0.40:
            return np.random.choice(self.responses[best_match_idx])

        # For factual queries, require stronger confidence before returning canned intents.
        if is_knowledge_query and (best_tag in small_talk_tags or best_score < 0.78 or has_extra_topic_tokens):
            return self.get_generative_response(llm_input, chat_history=chat_history)

        if best_score > 0.62:
            return np.random.choice(self.responses[best_match_idx])

        # Default to generative fallback for low-confidence matches.
        return self.get_generative_response(llm_input, chat_history=chat_history)

    def get_response(self, user_input, context_text=None, context_name=None, chat_history=None):
        """Given user input, find the best matching response."""
        quoted_prompts = self._extract_quoted_prompts(user_input)
        if quoted_prompts:
            answers = []
            for index, prompt in enumerate(quoted_prompts[:8], start=1):
                answer = self._get_single_response(prompt, context_text=context_text, context_name=context_name, chat_history=chat_history)
                answers.append(f"{index}. {answer}")
            return "\n\n".join(answers)

        questions = self._split_compound_questions(user_input)
        if len(questions) > 1:
            answers = []
            for index, question in enumerate(questions[:6], start=1):
                answer = self._get_single_response(question, context_text=context_text, context_name=context_name, chat_history=chat_history)
                answers.append(f"{index}. {answer}")
            return "\n\n".join(answers)

        return self._get_single_response(user_input, context_text=context_text, context_name=context_name, chat_history=chat_history)

# Example usage (for testing)
if __name__ == "__main__":
    bot = ChatBot()
    print("Chatbot initialized. Type 'quit' to exit.")
    while True:
        user_msg = input("You: ")
        if user_msg.lower() == 'quit':
            break
        print("Bot:", bot.get_response(user_msg))
