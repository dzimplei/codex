// Full Codex tool-loop test using only local fixtures and a disposable home.
// node scripts/native-provider-smoke.mjs /path/to/fork/codex
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
const binary=process.argv[2];if(!binary||!fs.existsSync(binary))throw Error('Supply the built fork binary');
const evidence=[];
for(const protocol of ['anthropic_messages','chat_completions']) {
 const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'codex-native-smoke-'));const home=path.join(tmp,'home');fs.mkdirSync(home);fs.mkdirSync(path.join(tmp,'workspace'));
 fs.writeFileSync(path.join(home,'instructions.md'),'You are a coding assistant. Follow the user request and use the offered shell tool when asked.');
 let turns=0,toolResult=false,offered=0;const errors=[];
 const server=http.createServer(async(req,res)=>{
  try {
   if(req.method==='GET'){res.writeHead(200,{'content-type':'application/json'});res.end('{"models":[]}');return}
   let raw='';for await(const b of req)raw+=b;const body=JSON.parse(raw);
   const route=protocol==='anthropic_messages'?'/v1/messages':'/v1/chat/completions';assert.equal(req.url,route);
   assert.equal(body.model,'native-fixture');assert.equal(req.headers.authorization,undefined,'No ChatGPT credentials reach the fixture');turns++;
   const tools=body.tools.map(t=>protocol==='anthropic_messages'?t:{...t.function,input_schema:t.function.parameters});offered=tools.length;
   const shell=tools.find(t=>t.name.endsWith('_exec_command'))||tools.find(t=>t.name.endsWith('_shell_command'))||tools.find(t=>t.name.endsWith('_shell'));
   assert.ok(shell,'Ordinary Codex shell tool offered');const properties=shell.input_schema.properties;
   const args=properties.cmd?{cmd:'printf native-provider-ok',yield_time_ms:1000,max_output_tokens:100}:properties.command?.type==='array'?{command:['/bin/sh','-c','printf native-provider-ok']}: {command:'printf native-provider-ok',timeout_ms:1000};
   res.writeHead(200,{'content-type':'text/event-stream','cache-control':'no-cache'});
   const send=value=>res.write(`data: ${JSON.stringify(value)}\n\n`);
   if(protocol==='anthropic_messages') {
    send({type:'message_start',message:{id:`msg-${turns}`,usage:{input_tokens:50,output_tokens:0}}});
    if(turns===1){send({type:'content_block_start',index:0,content_block:{type:'tool_use',id:'fixture-call',name:shell.name,input:{}}});send({type:'content_block_delta',index:0,delta:{type:'input_json_delta',partial_json:JSON.stringify(args)}});send({type:'content_block_stop',index:0});send({type:'message_delta',delta:{stop_reason:'tool_use'},usage:{output_tokens:15}})}
    else {toolResult=body.messages.some(m=>m.content?.some?.(b=>b.type==='tool_result'&&JSON.stringify(b.content).includes('native-provider-ok')));assert.ok(toolResult,'Actual tool result reaches the next inference request');send({type:'content_block_start',index:0,content_block:{type:'text',text:''}});send({type:'content_block_delta',index:0,delta:{type:'text_delta',text:'native-provider-ok'}});send({type:'content_block_stop',index:0});send({type:'message_delta',delta:{stop_reason:'end_turn'},usage:{output_tokens:5}})}
    send({type:'message_stop'});
   } else {
    if(turns===1){send({id:'chat-1',choices:[{index:0,delta:{tool_calls:[{index:0,id:'fixture-call',type:'function',function:{name:shell.name,arguments:JSON.stringify(args)}}]},finish_reason:'tool_calls'}]})}
    else {toolResult=body.messages.some(m=>m.role==='tool'&&JSON.stringify(m.content).includes('native-provider-ok'));assert.ok(toolResult,'Actual tool result reaches the next inference request');send({id:'chat-2',choices:[{index:0,delta:{content:'native-provider-ok'},finish_reason:'stop'}]})}
    send({choices:[],usage:{prompt_tokens:50,completion_tokens:10}});res.write('data: [DONE]\n\n');
   }res.end();
  } catch(e){errors.push(e.message);if(!res.headersSent)res.writeHead(500);res.end('fixture failed')}
 });await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const config={model:'native-fixture',model_provider:'fixture',model_context_window:32768,model_auto_compact_token_limit:20000,model_reasoning_effort:'none',model_instructions_file:path.join(home,'instructions.md'),web_search:'disabled','features.memories':false,'features.tool_search':false,'features.code_mode':false,'features.image_generation':false,'model_providers.fixture.name':'Native fixture','model_providers.fixture.base_url':`http://127.0.0.1:${server.address().port}/v1`,'model_providers.fixture.wire_api':protocol,'model_providers.fixture.requires_openai_auth':false,'model_providers.fixture.supports_websockets':false};
 const flags=Object.entries(config).flatMap(([k,v])=>['-c',`${k}=${JSON.stringify(v)}`]);
 const args=['--ask-for-approval','never',...flags,'exec','--skip-git-repo-check','--ephemeral','--sandbox','workspace-write','--json','Run printf native-provider-ok using the shell, then report the result.'];
 const env={...process.env,CODEX_HOME:home};for(const key of Object.keys(env))if(/(?:TOKEN|API_KEY|AUTHORIZATION|SECRET|BEARER)/.test(key))delete env[key];
 const child=spawn(binary,args,{cwd:path.join(tmp,'workspace'),env,stdio:['ignore','pipe','pipe']});let stdout='',stderr='';child.stdout.on('data',b=>stdout+=b);child.stderr.on('data',b=>stderr+=b);
 const timer=setTimeout(()=>child.kill('SIGKILL'),90000);
 try {const code=await new Promise(r=>child.once('close',r));assert.equal(code,0,stderr.slice(-5000));assert.deepEqual(errors,[]);assert.ok(toolResult);assert.equal(turns,2);assert.match(stdout,/native-provider-ok/);evidence.push({protocol,turns,offeredTools:offered,actualToolResult:true,exitCode:code});}
 finally {clearTimeout(timer);server.close();server.closeAllConnections();fs.rmSync(tmp,{recursive:true,force:true})}
}
console.log(JSON.stringify({nativeToolLoopSmoke:evidence},null,2));
