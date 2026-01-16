# What are the E2E user stories I want to support?

* I want to start over and rebuild the entire codebase with a few goals in mind:
  * simplify and reduce interdependencies as much as possible
  * everything should be built to be accessible to an agent that will help users build and maintain their knowledge systems. Each component will be an agentic Skill (in the Anthropic sense) that the agent can use.
* each separate "project" is a Knowledge Organization System (KOS).
* vocabulary to ontology are foundational KOS documents written with LinkML; each project has only one of each. However, these can support several graph artifacts within the same project:
  * atemporal knowledge graph
  * temporal knowledge graph
  * causal dag
  * process dag
  * reasoning map (for agentic workflows)
* each step of the KOS needs to identify sources, whether manually defined by user or extracted from a document. If the latter, then specific spans within the document are cited.



