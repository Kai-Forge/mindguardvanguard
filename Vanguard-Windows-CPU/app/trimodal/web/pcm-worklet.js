class PCMWindow extends AudioWorkletProcessor {
  constructor(){super();this.buffer=new Float32Array(Math.round(sampleRate*2));this.offset=0;this.startFrame=0;this.port.onmessage=e=>{if(e.data==='flush'&&this.offset){this.port.postMessage({samples:this.buffer.slice(0,this.offset),startFrame:this.startFrame,endFrame:this.startFrame+this.offset});this.offset=0;}};}
  process(inputs){const ch=inputs[0]?.[0];if(!ch)return true;let pos=0;while(pos<ch.length){if(this.offset===0)this.startFrame=currentFrame+pos;const n=Math.min(ch.length-pos,this.buffer.length-this.offset);this.buffer.set(ch.subarray(pos,pos+n),this.offset);this.offset+=n;pos+=n;if(this.offset===this.buffer.length){this.port.postMessage({samples:this.buffer.slice(),startFrame:this.startFrame,endFrame:this.startFrame+this.offset});this.offset=0;}}return true;}
}
registerProcessor('pcm-window',PCMWindow);
