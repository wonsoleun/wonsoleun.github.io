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
        description: "How models train, how they are built, and how they generalize, hands-on — optimizers, loss landscapes, CNN architectures, receptive fields, overfitting, and regularization.",
    },
];
