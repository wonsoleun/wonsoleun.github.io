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
        title: "Concepts of Machine · Deep Learning",
        description: "Training dynamics and generalization, hands-on — optimizers, loss landscapes, learning rates, overfitting, and regularization.",
    },
];
