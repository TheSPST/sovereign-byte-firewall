\# TECHNICAL SPECIFICATION: SOVEREIGN BYTE-LEVEL ANOMALY DETECTION ENGINE  
**\*\*Target Compute Environment:\*\*** AI Kosh (CDAC Airawat) / Multi-GPU NVIDIA A100 Cluster    
**\*\*Local Prototyping Environment:\*\*** macOS Apple Silicon (M2 Pro)

\---

\#\# 1\. Executive Summary & Problem Statement

\#\#\# The Problem  
Traditional enterprise cybersecurity relies heavily on **\*\*Deep Packet Inspection (DPI)\*\*** and text-based security logs (e.g., SIEMs, Syslogs). This introduces a critical architectural bottleneck:  
1\. **\*\*Data Inflation:\*\*** Translating raw binary packet streams into human-readable text logs expands data volume by up to 500%, creating severe processing latency.  
2\. **\*\*The "Patient Zero" Dependency:\*\*** Traditional firewalls depend on pre-compiled threat signatures. When a zero-day exploit or highly dynamic, AI-generated polymorphic malware hits a network, signature-based systems fail because they lack a text-based definition for the attack.  
3\. **\*\*Privacy & Exfiltration Risks:\*\*** Cloud-based AI security engines require shipping sensitive raw network traffic metadata off-site, violating strict zero-trust architectures and compliance frameworks (such as healthcare and defense).

\#\#\# The Solution: Encoder-Free Sovereign Intelligence  
This architecture bypasses text translation completely. By adapting a **\*\*Byte Latent Transformer (BLT)\*\*** framework, the system treats raw network traffic (\`.pcap\`) as a continuous 1D stream of raw integers ($0 \\text{ to } 255$). 

The model natively learns the mathematical "grammar" and structural "syntax" of network protocols at the byte level. Malicious zero-day payloads, obfuscated command-and-control heartbeats, and fuzzing attacks are detected instantly because they break the foundational entropy patterns of normal network traffic, allowing for completely offline, air-gapped threat mitigation.

\---

\#\# 2\. Component 1: High-Throughput Streaming Dataloader

To feed a Byte Latent Transformer on an A100 cluster without running out of CPU RAM, we implement a custom PyTorch \`IterableDataset\`. It utilizes \`scapy.utils.RawPcapReader\` to stream raw bytes off the disk packet-by-packet, enforcing strict sequence windowing to prevent VRAM overflow.

\`\`\`python  
import torch  
from torch.utils.data import IterableDataset, DataLoader  
from scapy.utils import RawPcapReader

class RawPcapIterableDataset(IterableDataset):  
    def \_\_init\_\_(self, pcap\_path, max\_sequence\_length=8192):  
        super().\_\_init\_\_()  
        self.pcap\_path \= pcap\_path  
        self.max\_sequence\_length \= max\_sequence\_length

    def \_\_iter\_\_(self):  
        try:  
            with RawPcapReader(self.pcap\_path) as pcap\_reader:  
                current\_sequence \= \[\]  
                  
                for packet\_data, \_metadata in pcap\_reader:  
                    \# Append raw byte integer values directly (0-255)  
                    for byte in packet\_data:  
                        current\_sequence.append(byte)  
                          
                        \# Once our chunk hits the sequence boundary, yield it  
                        if len(current\_sequence) \== self.max\_sequence\_length:  
                            yield torch.tensor(current\_sequence, dtype=torch.long)  
                            current\_sequence \= \[\]  
                  
                \# Yield any remaining trailing bytes padded with 0x00  
                if current\_sequence:  
                    remainder \= self.max\_sequence\_length \- len(current\_sequence)  
                    current\_sequence.extend(\[0\] \* remainder)  
                    yield torch.tensor(current\_sequence, dtype=torch.long)  
                      
        except FileNotFoundError:  
            raise FileNotFoundError(f"Target PCAP file not found at: {self.pcap\_path}")

\# Factory function to spin up the DataLoader on AI Kosh cluster  
def get\_pcap\_dataloader(pcap\_path, batch\_size=32, num\_workers=4):  
    dataset \= RawPcapIterableDataset(pcap\_path)  
    return DataLoader(dataset, batch\_size=batch\_size, num\_workers=num\_workers)

## **3\. Component 2: The Network-Byte Entropy Patcher**

### **Mathematical Framework**

To process raw byte streams efficiently, the system utilizes a **Predictive Next-Byte Network** as a patcher. The model operates over a sliding window to predict the probability distribution $P$ of the next incoming byte:  
$$P(b\_t \\mid b\_{t-1}, b\_{t-2}, \\dots, b\_{t-N})$$  
The cross-entropy loss $H$ at byte position $t$ is calculated using Shannon Entropy:  
$$H\_t \= \- \\sum\_{i=0}^{255} P(x\_i) \\log\_2 P(x\_i)$$  
Structured protocol headers yield low entropy ($H\_t \\to 0$), while randomized or malicious payloads cause an abrupt spike in entropy ($H\_t \\gg 0$). A patch boundary is declared whenever $H\_t \> \\tau$ (where $\\tau$ is the configured threshold).

### **PyTorch Implementation** 

import torch  
import torch.nn as nn  
import torch.nn.functional as F

class NetworkBytePatcher(nn.Module):  
    def \_\_init\_\_(self, d\_model=128, nhead=4, num\_layers=2, max\_patch\_size=64):  
        super().\_\_init\_\_()  
        self.vocab\_size \= 256  \# Byte value range 0x00 \- 0xFF  
        self.max\_patch\_size \= max\_patch\_size  
        self.embedding \= nn.Embedding(self.vocab\_size, d\_model)  
          
        encoder\_layer \= nn.TransformerEncoderLayer(  
            d\_model=d\_model, nhead=nhead, dim\_feedforward=d\_model \* 4, batch\_first=True  
        )  
        self.transformer \= nn.TransformerEncoder(encoder\_layer, num\_layers=num\_layers)  
        self.predictor \= nn.Linear(d\_model, self.vocab\_size)

    def forward(self, x):  
        seq\_len \= x.size(1)  
        causal\_mask \= nn.Transformer.generate\_square\_subsequent\_mask(seq\_len).to(x.device)  
        emb \= self.embedding(x)  
        out \= self.transformer(emb, mask=causal\_mask, is\_causal=True)  
        return self.predictor(out)

    def compute\_entropy(self, logits):  
        probs \= F.softmax(logits, dim=-1)  
        return \-torch.sum(probs \* torch.log2(probs \+ 1e-9), dim=-1)

    def generate\_patch\_lengths(self, x, entropy\_threshold=5.0):  
        self.eval()  
        with torch.no\_grad():  
            with torch.amp.autocast('cuda'):  
                logits \= self.forward(x)  
                entropies \= self.compute\_entropy(logits)  
          
        batch\_patch\_lengths \= \[\]  
        for b in range(x.size(0)):  
            current\_patch\_len \= 0  
            lengths \= \[\]  
            for t in range(x.size(1)):  
                current\_patch\_len \+= 1  
                if entropies\[b, t\] \> entropy\_threshold or current\_patch\_len \>= self.max\_patch\_size:  
                    lengths.append(current\_patch\_len)  
                    current\_patch\_len \= 0  
            if current\_patch\_len \> 0:  
                lengths.append(current\_patch\_len)  
            batch\_patch\_lengths.append(lengths)  
        return batch\_patch\_lengths

##  **4\. Component 3: AI Kosh Cluster Training & Checkpointing Wrapper** 

Because AI Kosh (CDAC Airawat) operates on time-limited SLURM execution slots, an un-interruptible checkpointing routine is non-negotiable. This wrapper ensures automatic state-saving and graceful job resumption.  
import os

def train\_patcher\_on\_kosh(model, dataloader, epochs=5, checkpoint\_dir="./checkpoints"):  
    os.makedirs(checkpoint\_dir, exist\_ok=True)  
    optimizer \= torch.optim.AdamW(model.parameters(), lr=1e-4)  
    criterion \= nn.CrossEntropyLoss()  
    scaler \= torch.amp.GradScaler('cuda')  
      
    start\_epoch \= 0  
    checkpoint\_path \= os.path.join(checkpoint\_dir, "latest\_patcher.pt")  
      
    \# Auto-resume logic if pre-empted by SLURM  
    if os.path.exists(checkpoint\_path):  
        print(f"Found active checkpoint. Resuming training on AI Kosh...")  
        checkpoint \= torch.load(checkpoint\_path)  
        model.load\_state\_dict(checkpoint\['model\_state'\])  
        optimizer.load\_state\_dict(checkpoint\['optimizer\_state'\])  
        start\_epoch \= checkpoint\['epoch'\]

    model.train()  
    model \= model.to('cuda')

    for epoch in range(start\_epoch, epochs):  
        for step, byte\_sequence in enumerate(dataloader):  
            byte\_sequence \= byte\_sequence.to('cuda')  
              
            inputs \= byte\_sequence\[:, :-1\]  
            targets \= byte\_sequence\[:, 1:\]  
              
            optimizer.zero\_grad()  
              
            with torch.amp.autocast('cuda'):  
                logits \= model(inputs)  
                loss \= criterion(logits.reshape(-1, 256), targets.reshape(-1))  
              
            scaler.scale(loss).backward()  
            scaler.step(optimizer)  
            scaler.update()  
              
            if step % 500 \== 0:  
                print(f"Epoch \[{epoch}/{epochs}\] | Step {step} | Loss: {loss.item():.4f}")  
                  
        \# Mandatory end-of-epoch state save for cluster durability  
        torch.save({  
            'epoch': epoch \+ 1,  
            'model\_state': model.state\_dict(),  
            'optimizer\_state': optimizer.state\_dict(),  
        }, checkpoint\_path)  
        print(f"Checkpoint successfully secured to persistent storage at Epoch {epoch \+ 1}.")

## **5\. Data Sourcing Strategy (PCAP Acquisition)**

To train this model on AI Kosh, developers must pull raw packet captures directly into the network cluster storage via command-line tools. The target public baselines include:

1. **CIC-IDS2017 & CIC-DDoS2019 (Canadian Institute for Cybersecurity):** Contains over 50GB of raw network execution labs containing real multi-day simulations of Botnets, DDoS, and Brute Force attacks.  
2. **MACCDC (Mid-Atlantic Collegiate Cyber Defense Competition via Netresec):** Live-fire chaotic adversarial capture data where professional red-teams actively attempt to compromise live systems.  
3. **CTU-13 (Stratosphere IPS Dataset):** Long-running tracking of encrypted, obfuscated command-and-control (C2) botnet heartbeats.

## **6\. Master Agentic AI Execution Prompt**

*Copy and paste the markdown block below directly into Cursor, Codex, or Continue to coordinate your agentic development workflows:*  
\# Role  
You are a Staff-Level Systems AI Architect specializing in PyTorch optimization, low-level network hooks (eBPF, scapy), and the Byte Latent Transformer (BLT) architecture.

\# Context  
We are deploying a foundational, encoder-free byte-level network anomaly detector on the AI Kosh cluster (NVIDIA A100 architecture). Our primary engineering task is to optimize the Network Entropy Patcher so it successfully generates dynamic sequence cuts on raw 0-255 integer packet arrays without memory leaks.

\# Immediate Instructions  
1\. Review the provided \`NetworkBytePatcher\` PyTorch architecture.  
2\. Optimize the multi-headed causal self-attention scaling function within the Transformer blocks using PyTorch's native \`F.scaled\_dot\_product\_attention\` to ensure maximum compatibility with both local macOS Apple Silicon MPS (for debugging) and NVIDIA CUDA (for A100 training).  
3\. Generate a robust evaluation script that takes the outputs from \`generate\_patch\_lengths\`, counts total tokens generated per network window, and visualizes the running entropy profile across an arbitrary packet. 

Prioritize structural clarity, zero-copy memory layouts, and clean type-hinting across all Python components.  
