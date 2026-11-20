// Playground 섹션 목록. 위에서부터 순서대로 표시돼요.
// 카드(md 파일)의 frontmatter `section` 값이 아래 id와 같아야 해당 섹션에 들어가요.
export interface PlaygroundSection {
    id: string;
    title: string;
    description?: string;
}

export const PLAYGROUND_SECTIONS: PlaygroundSection[] = [
    {
        id: "optimization",
        title: "Optimization & Training",
        description: "손실 지형 위에서 옵티마이저가 어떻게 움직이는지, 학습률과 하이퍼파라미터가 무엇을 바꾸는지 직접 확인해요.",
    },
];
