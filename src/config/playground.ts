// Playground sections, displayed top to bottom in this order.
// A card (md file) appears in a section when its frontmatter `section` matches that section's id.
export interface PlaygroundSection {
    id: string;
    title: string;
    description?: string;
}

export const PLAYGROUND_SECTIONS: PlaygroundSection[] = [
    {
        id: "optimization",
        title: "Optimization & Training",
        description: "See how optimizers move across a loss landscape, and what learning rates and other hyperparameters change.",
    },
];
