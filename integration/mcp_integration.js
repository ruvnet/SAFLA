/**
 * SAFLA MCP Integration Module (JavaScript)
 * 
 * This module provides a JavaScript interface for integrating with the SAFLA MCP server,
 * enabling applications to leverage SAFLA's meta-cognitive capabilities, goal management,
 * and performance optimization features.
 */

const { spawn } = require('child_process');
const { EventEmitter } = require('events');

class SAFLAMCPClient extends EventEmitter {
  /**
   * @param {string}   command  Executable to run (e.g. 'python'). Must not contain spaces.
   * @param {string[]} args     Argument list passed directly to spawn — never joined into a
   *                            shell string, preventing argument-injection attacks.
   */
  constructor(command = 'python', args = ['safla/mcp_stdio_server.py']) {
    super();
    // Validate that command does not contain shell metacharacters
    if (/[\s;&|`$<>\\'"!]/.test(command)) {
      throw new Error(`Invalid command: shell metacharacters are not allowed in command ('${command}')`);
    }
    this.command = command;
    this.args = args;
    this.process = null;
    this.messageId = 0;
    this.pendingRequests = new Map();
  }

  /**
   * Start the MCP server connection
   */
  async connect() {
    return new Promise((resolve, reject) => {
      try {
        this.process = spawn(this.command, this.args, {
          stdio: ['pipe', 'pipe', 'pipe']
        });

        if (this.process.stdout) {
          this.process.stdout.on('data', (data) => {
            const lines = data.toString().split('\n').filter(line => line.trim());
            for (const line of lines) {
              try {
                const message = JSON.parse(line);
                this.handleMessage(message);
              } catch (error) {
                console.error('Failed to parse MCP message:', line, error);
              }
            }
          });
        }

        if (this.process.stderr) {
          this.process.stderr.on('data', (data) => {
            console.error('MCP Server Error:', data.toString());
          });
        }

        if (this.process) {
          this.process.on('error', (error) => {
            console.error('MCP Process Error:', error);
            reject(error);
          });

          this.process.on('exit', (code) => {
            console.log('MCP Server exited with code:', code);
            this.emit('disconnect');
          });
        }

        // Initialize connection
        this.sendRequest('initialize', {
          protocolVersion: '2024-11-05',
          capabilities: {},
          clientInfo: {
            name: 'SAFLA-MCP-Client',
            version: '1.0.0'
          }
        }).then(() => {
          resolve();
        }).catch(reject);

      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the MCP server
   */
  disconnect() {
    if (this.process) {
      this.process.kill();
      this.process = null;
    }
  }

  /**
   * Send a request to the MCP server
   */
  async sendRequest(method, params) {
    return new Promise((resolve, reject) => {
      if (!this.process || !this.process.stdin) {
        reject(new Error('MCP server not connected'));
        return;
      }

      const id = ++this.messageId;
      const message = {
        jsonrpc: '2.0',
        id,
        method,
        params
      };

      this.pendingRequests.set(id, { resolve, reject });

      try {
        this.process.stdin.write(JSON.stringify(message) + '\n');
      } catch (error) {
        this.pendingRequests.delete(id);
        reject(error);
      }
    });
  }

  /**
   * Handle incoming messages from the MCP server
   */
  handleMessage(message) {
    if (message.id !== undefined && this.pendingRequests.has(message.id)) {
      const { resolve, reject } = this.pendingRequests.get(message.id);
      this.pendingRequests.delete(message.id);

      if (message.error) {
        reject(new Error(`MCP Error ${message.error.code}: ${message.error.message}`));
      } else {
        resolve(message.result);
      }
    } else if (message.method) {
      // Handle notifications
      this.emit('notification', message.method, message.params);
    }
  }

  /**
   * Call an MCP tool
   */
  async callTool(name, arguments_ = {}) {
    return this.sendRequest('tools/call', {
      name,
      arguments: arguments_
    });
  }

  // Meta-Cognitive Engine Methods

  /**
   * Get current system awareness state
   */
  async getSystemAwareness() {
    const result = await this.callTool('get_system_awareness');
    return result.awareness_state;
  }

  /**
   * Perform introspective analysis
   */
  async analyzeIntrospection(analysisType = 'comprehensive', timeWindowHours = 24) {
    return this.callTool('analyze_system_introspection', {
      analysis_type: analysisType,
      time_window_hours: timeWindowHours
    });
  }

  /**
   * Update system awareness parameters
   */
  async updateAwareness(awarenessLevel, focusAreas, introspectionDepth) {
    const params = {};
    if (awarenessLevel !== undefined) params.awareness_level = awarenessLevel;
    if (focusAreas) params.focus_areas = focusAreas;
    if (introspectionDepth) params.introspection_depth = introspectionDepth;

    return this.callTool('update_awareness_state', params);
  }

  // Goal Management Methods

  /**
   * Create a new goal
   */
  async createGoal(name, description, priority = 'medium', targetMetrics, deadline) {
    const params = {
      goal_name: name,
      description,
      priority
    };
    if (targetMetrics) params.target_metrics = targetMetrics;
    if (deadline) params.deadline = deadline;

    const result = await this.callTool('create_goal', params);
    return result.goal;
  }

  /**
   * List all goals
   */
  async listGoals(statusFilter = 'all', priorityFilter = 'all') {
    const result = await this.callTool('list_goals', {
      status_filter: statusFilter,
      priority_filter: priorityFilter
    });
    return result.goals;
  }

  /**
   * Update a goal
   */
  async updateGoal(goalId, updates) {
    return this.callTool('update_goal', {
      goal_id: goalId,
      ...updates
    });
  }

  /**
   * Evaluate goal progress
   */
  async evaluateGoalProgress(goalId) {
    const params = { include_recommendations: true };
    if (goalId) params.goal_id = goalId;

    return this.callTool('evaluate_goal_progress', params);
  }

  // Performance and Benchmarking Methods

  /**
   * Benchmark memory performance
   */
  async benchmarkMemory(testDuration = 60, memoryPatterns = ['sequential', 'random']) {
    return this.callTool('benchmark_memory_performance', {
      test_duration: testDuration,
      memory_patterns: memoryPatterns
    });
  }

  /**
   * Benchmark vector operations
   */
  async benchmarkVectorOperations(vectorCount = 1000, vectorDimensions = 512, operations = ['similarity', 'storage', 'retrieval']) {
    return this.callTool('benchmark_vector_operations', {
      vector_count: vectorCount,
      vector_dimensions: vectorDimensions,
      operations
    });
  }

  /**
   * Test MCP connectivity
   */
  async testConnectivity(testDepth = 'basic') {
    return this.callTool('test_mcp_connectivity', {
      test_depth: testDepth
    });
  }

  // System Information Methods

  /**
   * Validate SAFLA installation
   */
  async validateInstallation() {
    return this.callTool('validate_installation');
  }

  /**
   * Get system information
   */
  async getSystemInfo() {
    return this.callTool('get_system_info');
  }

  /**
   * Get configuration summary
   */
  async getConfigSummary() {
    return this.callTool('get_config_summary');
  }

  /**
   * Check GPU status
   */
  async checkGPUStatus() {
    return this.callTool('check_gpu_status');
  }
}

// Example usage function
async function demonstrateMCPIntegration() {
  const client = new SAFLAMCPClient();

  try {
    console.log('Connecting to SAFLA MCP server...');
    await client.connect();

    console.log('Getting system awareness...');
    const awareness = await client.getSystemAwareness();
    console.log('System Awareness:', JSON.stringify(awareness, null, 2));

    console.log('Creating a test goal...');
    const goal = await client.createGoal(
      'Performance Optimization',
      'Optimize system performance for better response times',
      'high',
      { target_response_time: 100, target_efficiency: 0.95 }
    );
    console.log('Created Goal:', JSON.stringify(goal, null, 2));

    console.log('Running memory benchmark...');
    const memoryBenchmark = await client.benchmarkMemory(30, ['sequential']);
    console.log('Memory Benchmark:', JSON.stringify(memoryBenchmark, null, 2));

    console.log('Testing connectivity...');
    const connectivity = await client.testConnectivity('comprehensive');
    console.log('Connectivity Test:', JSON.stringify(connectivity, null, 2));

  } catch (error) {
    console.error('MCP Integration Error:', error);
  } finally {
    client.disconnect();
  }
}

module.exports = {
  SAFLAMCPClient,
  demonstrateMCPIntegration
};

// If running directly, demonstrate the integration
if (require.main === module) {
  demonstrateMCPIntegration();
}