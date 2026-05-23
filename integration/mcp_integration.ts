/**
 * SAFLA MCP Integration Module
 * 
 * This module provides a TypeScript interface for integrating with the SAFLA MCP server,
 * enabling applications to leverage SAFLA's meta-cognitive capabilities, goal management,
 * and performance optimization features.
 */

// For Node.js environments
declare const require: any;
const { spawn } = require('child_process');
const { EventEmitter } = require('events');

// Type definitions for Node.js modules
interface ChildProcess {
  stdout?: any;
  stderr?: any;
  stdin?: any;
  kill(): void;
  on(event: string, listener: (...args: any[]) => void): void;
}

export interface MCPMessage {
  jsonrpc: string;
  id?: string | number;
  method?: string;
  params?: any;
  result?: any;
  error?: {
    code: number;
    message: string;
    data?: any;
  };
}

export interface SystemAwareness {
  awareness_level: number;
  focus_areas: string[];
  introspection_depth: 'shallow' | 'moderate' | 'deep';
  system_metrics: {
    uptime_hours: number;
    active_sessions: number;
    performance_score: number;
    load_factor: number;
    memory_efficiency: number;
  };
  self_assessment: {
    confidence: number;
    competence: number;
    adaptability: number;
  };
  last_introspection: number;
  timestamp: number;
}

export interface Goal {
  id: string;
  type?: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  target_value?: number;
  current_value: number;
  progress: number;
  status: 'active' | 'completed' | 'paused' | 'failed';
  created_at: number;
  deadline?: number;
  milestones: any[];
  metrics: {
    attempts: number;
    successes: number;
    failures: number;
  };
}

export interface BenchmarkResult {
  benchmark_id: string;
  timestamp: number;
  results: Record<string, any>;
  system_info?: Record<string, any>;
}

export class SAFLAMCPClient extends EventEmitter {
  private process: ChildProcess | null = null;
  private messageId = 0;
  private pendingRequests = new Map<string | number, {
    resolve: (value: any) => void;
    reject: (error: any) => void;
  }>();

  /**
   * @param command  Executable to run (e.g. 'python'). Must not contain spaces.
   * @param args     Argument list passed directly to spawn — never joined into a
   *                 shell string, preventing argument-injection attacks.
   */
  constructor(
    private command: string = 'python',
    private args: string[] = ['safla/mcp_stdio_server.py']
  ) {
    super();
    // Validate that command does not contain shell metacharacters
    if (/[\s;&|`$<>\\'"!]/.test(command)) {
      throw new Error(`Invalid command: shell metacharacters are not allowed in command ('${command}')`);
    }
  }

  /**
   * Start the MCP server connection
   */
  async connect(): Promise<void> {
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
                const message: MCPMessage = JSON.parse(line);
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
  disconnect(): void {
    if (this.process) {
      this.process.kill();
      this.process = null;
    }
  }

  /**
   * Send a request to the MCP server
   */
  private async sendRequest(method: string, params?: any): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.process || !this.process.stdin) {
        reject(new Error('MCP server not connected'));
        return;
      }

      const id = ++this.messageId;
      const message: MCPMessage = {
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
  private handleMessage(message: MCPMessage): void {
    if (message.id !== undefined && this.pendingRequests.has(message.id)) {
      const { resolve, reject } = this.pendingRequests.get(message.id)!;
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
  async callTool(name: string, arguments_: any = {}): Promise<any> {
    return this.sendRequest('tools/call', {
      name,
      arguments: arguments_
    });
  }

  // Meta-Cognitive Engine Methods

  /**
   * Get current system awareness state
   */
  async getSystemAwareness(): Promise<SystemAwareness> {
    const result = await this.callTool('get_system_awareness');
    return result.awareness_state;
  }

  /**
   * Perform introspective analysis
   */
  async analyzeIntrospection(
    analysisType: 'performance' | 'behavior' | 'goals' | 'comprehensive' = 'comprehensive',
    timeWindowHours: number = 24
  ): Promise<any> {
    return this.callTool('analyze_system_introspection', {
      analysis_type: analysisType,
      time_window_hours: timeWindowHours
    });
  }

  /**
   * Update system awareness parameters
   */
  async updateAwareness(
    awarenessLevel?: number,
    focusAreas?: string[],
    introspectionDepth?: 'shallow' | 'moderate' | 'deep'
  ): Promise<any> {
    const params: any = {};
    if (awarenessLevel !== undefined) params.awareness_level = awarenessLevel;
    if (focusAreas) params.focus_areas = focusAreas;
    if (introspectionDepth) params.introspection_depth = introspectionDepth;

    return this.callTool('update_awareness_state', params);
  }

  // Goal Management Methods

  /**
   * Create a new goal
   */
  async createGoal(
    name: string,
    description: string,
    priority: 'low' | 'medium' | 'high' | 'critical' = 'medium',
    targetMetrics?: Record<string, any>,
    deadline?: number
  ): Promise<Goal> {
    const params: any = {
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
  async listGoals(
    statusFilter: 'all' | 'active' | 'completed' | 'paused' | 'failed' = 'all',
    priorityFilter: 'all' | 'low' | 'medium' | 'high' | 'critical' = 'all'
  ): Promise<Goal[]> {
    const result = await this.callTool('list_goals', {
      status_filter: statusFilter,
      priority_filter: priorityFilter
    });
    return result.goals;
  }

  /**
   * Update a goal
   */
  async updateGoal(
    goalId: string,
    updates: {
      status?: 'active' | 'completed' | 'paused' | 'failed';
      priority?: 'low' | 'medium' | 'high' | 'critical';
      progress?: number;
      notes?: string;
    }
  ): Promise<any> {
    return this.callTool('update_goal', {
      goal_id: goalId,
      ...updates
    });
  }

  /**
   * Evaluate goal progress
   */
  async evaluateGoalProgress(goalId?: string): Promise<any> {
    const params: any = { include_recommendations: true };
    if (goalId) params.goal_id = goalId;

    return this.callTool('evaluate_goal_progress', params);
  }

  // Performance and Benchmarking Methods

  /**
   * Benchmark memory performance
   */
  async benchmarkMemory(
    testDuration: number = 60,
    memoryPatterns: string[] = ['sequential', 'random']
  ): Promise<BenchmarkResult> {
    return this.callTool('benchmark_memory_performance', {
      test_duration: testDuration,
      memory_patterns: memoryPatterns
    });
  }

  /**
   * Benchmark vector operations
   */
  async benchmarkVectorOperations(
    vectorCount: number = 1000,
    vectorDimensions: number = 512,
    operations: string[] = ['similarity', 'storage', 'retrieval']
  ): Promise<BenchmarkResult> {
    return this.callTool('benchmark_vector_operations', {
      vector_count: vectorCount,
      vector_dimensions: vectorDimensions,
      operations
    });
  }

  /**
   * Test MCP connectivity
   */
  async testConnectivity(testDepth: 'basic' | 'comprehensive' = 'basic'): Promise<any> {
    return this.callTool('test_mcp_connectivity', {
      test_depth: testDepth
    });
  }

  // System Information Methods

  /**
   * Validate SAFLA installation
   */
  async validateInstallation(): Promise<any> {
    return this.callTool('validate_installation');
  }

  /**
   * Get system information
   */
  async getSystemInfo(): Promise<any> {
    return this.callTool('get_system_info');
  }

  /**
   * Get configuration summary
   */
  async getConfigSummary(): Promise<any> {
    return this.callTool('get_config_summary');
  }

  /**
   * Check GPU status
   */
  async checkGPUStatus(): Promise<any> {
    return this.callTool('check_gpu_status');
  }
}

// Example usage
export async function demonstrateMCPIntegration(): Promise<void> {
  const client = new SAFLAMCPClient();

  try {
    console.log('Connecting to SAFLA MCP server...');
    await client.connect();

    console.log('Getting system awareness...');
    const awareness = await client.getSystemAwareness();
    console.log('System Awareness:', awareness);

    console.log('Creating a test goal...');
    const goal = await client.createGoal(
      'Performance Optimization',
      'Optimize system performance for better response times',
      'high',
      { target_response_time: 100, target_efficiency: 0.95 }
    );
    console.log('Created Goal:', goal);

    console.log('Running memory benchmark...');
    const memoryBenchmark = await client.benchmarkMemory(30, ['sequential']);
    console.log('Memory Benchmark:', memoryBenchmark);

    console.log('Testing connectivity...');
    const connectivity = await client.testConnectivity('comprehensive');
    console.log('Connectivity Test:', connectivity);

  } catch (error) {
    console.error('MCP Integration Error:', error);
  } finally {
    client.disconnect();
  }
}

export default SAFLAMCPClient;