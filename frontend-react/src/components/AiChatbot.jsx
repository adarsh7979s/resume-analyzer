import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageCircle, X, Send, Bot, User, Sparkles } from 'lucide-react';
import './AiChatbot.css';

/* ── Scripted AI responses based on app state ── */
function getContextualResponses(state) {
  const { resumeUploaded, roleAnalyzed, score, candidateName, role, matched, missing, recommendations } = state;
  const name = candidateName?.trim() || 'there';
  const hasResults = score !== null;

  const base = [
    { trigger: /^(hi|hey|hello|hola)/i, response: `Hey ${name}! 👋 I'm your AI career assistant. How can I help you today? Try asking about resume tips, interview prep, or your analysis results!` },
    { trigger: /help|what can you do|commands/i, response: `Here's what I can help with:\n\n🔹 **Resume tips** — formatting & content advice\n🔹 **Interview prep** — common questions & strategies\n🔹 **Skill advice** — what to learn next\n🔹 **Analysis guide** — how to use this tool\n🔹 **Career strategy** — planning your next move\n\nJust type your question!` },
    { trigger: /how.*(use|work|start)/i, response: `Here's how to get started:\n\n1️⃣ **Upload** your resume PDF on the Dashboard\n2️⃣ **Select** a target role (e.g., "AI Engineer")\n3️⃣ **Click Analyze** to run the AI assessment\n4️⃣ Check your **Match Score** and **Skill Gap**\n5️⃣ Visit **Insights** for detailed recommendations\n\nThe whole process takes about 30 seconds! ⚡` },
    { trigger: /resume.*(tip|advice|improve|better|format)/i, response: `📝 **Resume Best Practices:**\n\n• Use a **single-column** layout for ATS compatibility\n• Start bullets with **action verbs** (Built, Led, Designed)\n• Include **quantified results** (e.g., "Increased sales by 30%")\n• Keep it to **1-2 pages** max\n• Use **keywords** from job descriptions\n• Add a clear **summary section** at the top\n• Remove photos, graphics, and fancy formatting` },
    { trigger: /interview|prepare|question/i, response: `🎯 **Interview Prep Tips:**\n\n• Research the company's **mission & recent news**\n• Prepare **STAR method** stories (Situation, Task, Action, Result)\n• Practice explaining your **projects** in 2 minutes\n• Prepare questions to **ask the interviewer**\n• For tech roles: review **system design** & coding fundamentals\n• Dress code: Match the **company culture** (when in doubt, business casual)` },
    { trigger: /ats|applicant track/i, response: `🤖 **ATS Optimization Tips:**\n\n• Use **standard section headers** (Experience, Education, Skills)\n• Avoid tables, columns, and complex formatting\n• Include **exact keywords** from the job posting\n• Use a **.docx or .pdf** format (not image-based)\n• Don't use headers/footers for critical information\n• Spell out abbreviations at least once` },
    { trigger: /salary|pay|negotiate|compensation/i, response: `💰 **Salary Negotiation Tips:**\n\n• Research market rates on **Glassdoor** and **Levels.fyi**\n• Never give a number first — ask for their range\n• Consider **total compensation** (equity, benefits, bonuses)\n• Practice saying: "Based on my research and experience, I'm targeting [range]"\n• Always **negotiate** — most offers have 10-20% flexibility` },
    { trigger: /linkedin|network|connect/i, response: `🔗 **LinkedIn & Networking Tips:**\n\n• Add a **professional headline** beyond just your job title\n• Write a compelling **About section** with keywords\n• Request recommendations from **former managers**\n• Engage with content in your **target industry**\n• Send personalized **connection requests** (mention shared interests)\n• Post about your **projects and learnings**` },
  ];

  // State-specific responses
  if (!resumeUploaded) {
    base.push({
      trigger: /^(what|where|how|next|status|score|result|analyz)/i,
      response: `You haven't uploaded a resume yet! Head to the **Dashboard** tab and:\n\n1. Drag & drop your resume PDF\n2. Click **Upload & Extract**\n3. Then choose a target role\n\nI'll give you personalized tips once your analysis is ready! 🚀`,
    });
  } else if (!hasResults) {
    base.push({
      trigger: /^(what|where|how|next|status|score|result|analyz)/i,
      response: `Great — your resume is uploaded! Now:\n\n1. Enter your **target role** (e.g., "Frontend Developer")\n2. Click **Analyze**\n3. Wait ~30 seconds for the AI assessment\n\nI'll have personalized recommendations once the analysis completes! 📊`,
    });
  }

  if (hasResults) {
    base.push(
      {
        trigger: /score|result|how.*(do|did)|performance/i,
        response: `📊 **Your Results:**\n\n• **Match Score:** ${score}%${score >= 80 ? ' — Excellent! 🎉' : score >= 60 ? ' — Strong foundation! 💪' : ' — Room for growth 📈'}\n• **Matched Skills:** ${matched?.length || 0}\n• **Skills to Develop:** ${missing?.length || 0}\n\n${score >= 70 ? 'You\'re in great shape for applications!' : 'Focus on closing your top skill gaps first.'} Check the **Insights** tab for your full action plan.`,
      },
      {
        trigger: /skill|learn|gap|missing|improve/i,
        response: missing?.length > 0
          ? `🎯 **Your Top Skill Gaps:**\n\n${missing.slice(0, 5).map((s, i) => `${i + 1}. **${s}**`).join('\n')}\n\nI'd recommend focusing on **${missing[0]}** first — it's likely the highest-impact skill for ${role || 'your target role'}. Check the Academy tab for course suggestions!`
          : `✨ Amazing! You have **zero skill gaps** for this role. Focus on:\n\n• Building portfolio projects\n• Strengthening your weakest matched skills\n• Preparing for behavioral interviews`,
      },
      {
        trigger: /strength|strong|good at|best/i,
        response: matched?.length > 0
          ? `💪 **Your Top Strengths:**\n\n${matched.slice(0, 5).map((m, i) => `${i + 1}. **${m.job_skill || m}**${m.similarity ? ` (${Math.round(m.similarity * 100)}%)` : ''}`).join('\n')}\n\nLeverage these in your resume summary and interview answers!`
          : `Upload and analyze your resume to discover your strengths! 📋`,
      },
    );

    if (recommendations?.action_plan?.length > 0) {
      base.push({
        trigger: /plan|action|what.*next|strategy|recommend/i,
        response: `📋 **Your Action Plan:**\n\n${recommendations.action_plan.slice(0, 3).map((a, i) => `${i + 1}. ${a}`).join('\n')}\n\nStart with step 1 and re-analyze after making improvements!`,
      });
    }
  }

  return base;
}

function findResponse(input, responses) {
  for (const r of responses) {
    if (r.trigger.test(input)) return r.response;
  }
  return null;
}

const QUICK_ACTIONS = [
  '👋 Hello!',
  '📝 Resume tips',
  '🎯 How to start',
  '💼 Interview prep',
];

export default function AiChatbot({ state }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const responses = useMemo(() => getContextualResponses(state), [
    state.resumeUploaded, state.roleAnalyzed, state.score,
    state.candidateName, state.role, state.matched, state.missing,
    state.recommendations,
  ]);

  // Initial greeting
  useEffect(() => {
    if (messages.length === 0) {
      const name = state.candidateName?.trim() || 'there';
      setMessages([{
        id: 'welcome',
        role: 'bot',
        text: `Hey ${name}! 👋 I'm your AI career assistant.\n\nI can help with resume tips, interview prep, skill advice, and guide you through the analysis process.\n\nTry one of the quick actions below, or ask me anything!`,
        time: new Date(),
      }]);
    }
  }, [state.candidateName]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input on open
  useEffect(() => {
    if (isOpen) inputRef.current?.focus();
  }, [isOpen]);

  function sendMessage(text) {
    if (!text.trim()) return;

    const userMsg = {
      id: Date.now() + '-u',
      role: 'user',
      text: text.trim(),
      time: new Date(),
    };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    // Find contextual response
    const reply = findResponse(text.trim(), responses);
    const fallback = `I'm not sure about that, but I can help with:\n\n• **Resume tips** — how to improve your resume\n• **Interview prep** — get ready for interviews\n• **Skill advice** — what to learn next\n• **How to start** — using the analyzer\n• **My score** — your analysis results\n\nTry asking about any of these! 😊`;

    // Simulate typing delay
    setTimeout(() => {
      setIsTyping(false);
      setMessages(prev => [...prev, {
        id: Date.now() + '-b',
        role: 'bot',
        text: reply || fallback,
        time: new Date(),
      }]);
    }, 600 + Math.random() * 800);
  }

  function handleSubmit(e) {
    e.preventDefault();
    sendMessage(input);
  }

  // Dynamic quick actions based on state
  const dynamicQuickActions = useMemo(() => {
    const actions = [...QUICK_ACTIONS];
    if (state.score !== null) {
      actions.push('📊 My score');
      actions.push('🎯 Skill gaps');
      actions.push('📋 Action plan');
    }
    return actions.slice(0, 6);
  }, [state.score]);

  return (
    <>
      {/* ── Floating trigger ── */}
      <AnimatePresence>
        {!isOpen && (
          <motion.button
            className="chatbot-trigger"
            onClick={() => setIsOpen(true)}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.92 }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
          >
            <MessageCircle size={22} />
            <span className="chatbot-trigger-pulse" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* ── Chat panel ── */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="chatbot-panel"
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ type: 'spring', stiffness: 300, damping: 25 }}
          >
            {/* Header */}
            <div className="chatbot-header">
              <div className="chatbot-header-left">
                <div className="chatbot-avatar">
                  <Bot size={16} />
                </div>
                <div>
                  <span className="chatbot-name">AI Assistant</span>
                  <span className="chatbot-status">
                    <span className="chatbot-status-dot" /> Online
                  </span>
                </div>
              </div>
              <button className="chatbot-close" onClick={() => setIsOpen(false)}>
                <X size={16} />
              </button>
            </div>

            {/* Messages */}
            <div className="chatbot-messages">
              {messages.map(msg => (
                <div key={msg.id} className={`chatbot-msg chatbot-msg-${msg.role}`}>
                  <div className="chatbot-msg-avatar">
                    {msg.role === 'bot' ? <Bot size={12} /> : <User size={12} />}
                  </div>
                  <div className="chatbot-msg-bubble">
                    {msg.text.split('\n').map((line, i) => (
                      <React.Fragment key={i}>
                        {line
                          .replace(/\*\*(.*?)\*\*/g, '⟨b⟩$1⟨/b⟩')
                          .split(/⟨\/?b⟩/)
                          .map((part, j) =>
                            j % 2 === 1 ? <strong key={j}>{part}</strong> : part
                          )}
                        {i < msg.text.split('\n').length - 1 && <br />}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              ))}

              {/* Typing indicator */}
              {isTyping && (
                <div className="chatbot-msg chatbot-msg-bot">
                  <div className="chatbot-msg-avatar"><Bot size={12} /></div>
                  <div className="chatbot-msg-bubble chatbot-typing">
                    <span /><span /><span />
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Quick actions */}
            <div className="chatbot-quick">
              {dynamicQuickActions.map(action => (
                <button
                  key={action}
                  className="chatbot-quick-btn"
                  onClick={() => sendMessage(action)}
                >
                  {action}
                </button>
              ))}
            </div>

            {/* Input */}
            <form className="chatbot-input-area" onSubmit={handleSubmit}>
              <input
                ref={inputRef}
                className="chatbot-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask me anything..."
              />
              <button type="submit" className="chatbot-send" disabled={!input.trim()}>
                <Send size={16} />
              </button>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
